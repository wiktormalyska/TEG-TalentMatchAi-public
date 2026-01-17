# data/data-loader.py
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document as LangChainDocument

# ------------------------------------------------------------------------------
# Konfiguracja Środowiska i Bibliotek
# Inicjalizacja silnika PDF, zmiennych środowiskowych i połączenia Neo4j
# ------------------------------------------------------------------------------
try:
    import pdfplumber
    PDF_ENGINE = "pdfplumber"
    print("Using PDF_ENGINE:", PDF_ENGINE)
except ImportError:
    PDF_ENGINE = "pypdf2"
    from PyPDF2 import PdfReader
    print("Using PDF_ENGINE:", PDF_ENGINE)

current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
project_root = current_dir.parent
env_path = project_root / '.env'

print(f"Loading .env from: {env_path.resolve()}")
load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
    raise RuntimeError("Missing Neo4j credentials in `'.env'`: set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ------------------------------------------------------------------------------
# Funkcje Pomocnicze i Dane Statyczne
# Obsługa plików PDF oraz ładowanie bazy umiejętności
# ------------------------------------------------------------------------------
def extract_text_from_pdf(path: Path) -> List[str]:
    pages = []
    if PDF_ENGINE == "pdfplumber":
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    else:
        reader = PdfReader(str(path))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return pages

def load_skills(filepath='skills-db.txt'):
    path = Path(filepath)
    if not path.exists():
        print(f"Warning: {filepath} not found. Using empty set.")
        return set()

    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

SKILLS_DB = load_skills('skills-db.txt')
print(f"Loaded {len(SKILLS_DB)} skills from file.")
known_skills_list = ", ".join(sorted(list(SKILLS_DB)))
print(f"Loaded :{known_skills_list}")

# ------------------------------------------------------------------------------
# Modele Danych i Konfiguracja AI
# Definicje schematów Pydantic oraz inicjalizacja klienta Azure OpenAI
# ------------------------------------------------------------------------------
class SkillAssessment(BaseModel):
    name: str = Field(
        description="Nazwa umiejętności dokładnie tak, jak występuje na podanej liście znanych skilli."
    )
    level: str = Field(
        description="Poziom biegłości oceniony na podstawie CV: 'Basic', 'Intermediate', 'Advanced', 'Expert'."
    )

class CVAnalysis(BaseModel):
    candidate_name: str = Field(description="Imię i nazwisko kandydata znalezione w CV.")
    seniority: str = Field(description="Ogólny poziom kandydata: Junior, Mid, Senior, Lead.")
    summary: str = Field(description="Krótkie podsumowanie kandydata (po polsku).")
    assessed_skills: List[SkillAssessment] = Field(
        description="Lista umiejętności znalezionych w CV, które pasują do dostarczonej listy skilli."
    )
    companies: List[str] = Field(
        description="Lista nazw firm (pracodawców). Ujednolić nazwy (np. 'Google Inc.' -> 'Google').")
    projects: List[str] = Field(description="Lista nazw kluczowych projektów.")
    universities: List[str] = Field(description="Lista nazw uczelni wyższych.")
    certifications: List[str] = Field(
        description="Lista nazw OFICJALNYCH certyfikatów branżowych (np. AWS Certified, ISTQB, Cisco CCNA). Ignoruj zwykłe szkolenia, kursy online bez egzaminu oraz opisy doświadczenia zawodowego."
    )

deploy_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_ver = os.getenv("AZURE_OPENAI_API_VERSION")

if not deploy_name or not api_ver:
    print("Warning: Missing Azure config in .env, using defaults.")
    deploy_name = "gpt-5-nano"
    api_ver = "2025-01-01-preview"

print(f"Using Azure Deployment: {deploy_name}, API Version: {api_ver}")

llm = AzureChatOpenAI(
    azure_deployment=deploy_name,
    api_version=api_ver,
)

cv_extractor = llm.with_structured_output(CVAnalysis)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Skill", "Company", "Project", "Certification", "University"],
    allowed_relationships=["HAS_SKILL", "WORKED_AT", "EARNED", "STUDIED_AT"]
)

# ------------------------------------------------------------------------------
# Logika Analizy AI
# Funkcje przetwarzające tekst CV na struktury danych
# ------------------------------------------------------------------------------
def analyze_cv_guided(text: str) -> CVAnalysis:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        Jesteś bardzo dokładnym i krytycznym analitykiem CV w rekrutacji IT. Twoim absolutnym priorytetem jest wierność faktom zawartym w tekście.
        NIE WOLNO CI ZMYŚLAĆ, ZGADYWAĆ ANI HALUCYNOWAĆ INFORMACJI, których nie ma wprost w dokumencie.

        Oto oficjalna lista umiejętności (SKILLS_DB), które śledzimy w systemie:
        {skills_list}

        Twoje zadania (przestrzegaj ich rygorystycznie):
        1. UMIEJĘTNOŚCI: Zidentyfikuj tylko te z powyższej listy, które kandydat faktycznie posiada. Mapuj synonimy do nazw z listy. Oceń poziom.
        2. FIRMY: Wyciągnij nazwy firm. Ujednolić nazwy (usuń "Sp. z o.o.", "Inc.").
        3. CERTYFIKATY (Krytyczne!):
           - Wyciągaj TYLKO oficjalne certyfikaty z konkretnymi nazwami (np. AWS Certified Solutions Architect, Cisco CCNA).
           - NIE wpisuj tu rzeczy typu "doświadczenie w Javie", "ukończony kurs Udemy" (chyba że to oficjalny certyfikat), "znajomość SQL".
           - Jeśli kandydat pisze, że "przygotowuje się do certyfikatu", TO NIE JEST CERTYFIKAT.
           - Jeśli brak wyraźnych certyfikatów, zostaw listę pustą.
        4. UCZELNIE: Tylko nazwy szkół wyższych.
        """),
        ("human", "{cv_text}")
    ])

    safe_text = text[:15000]

    prompt = prompt_template.invoke({
        "skills_list": known_skills_list,
        "cv_text": safe_text
    })

    try:
        return cv_extractor.invoke(prompt)
    except Exception as e:
        print(f"Błąd AI: {e}")
        return CVAnalysis(
            candidate_name="Unknown",
            seniority="Unknown",
            summary="Error analyzing file",
            assessed_skills=[],
            companies=[],
            projects=[],
            universities=[],
            certifications=[]
        )

# ------------------------------------------------------------------------------
# Operacje Bazodanowe Neo4j
# Funkcje tworzące węzły, relacje i więzy integralności
# ------------------------------------------------------------------------------
ENTITY_LABELS = ["Person", "Skill", "Company", "Project", "Certification", "University"]

def setup_database_constraints(driver):
    print("  -> Setting up database constraints for entities...")
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE")
        for label in ENTITY_LABELS:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{label}) REQUIRE e.id IS UNIQUE")

def clear_database(driver):
    print("  -> Clearing database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  -> Database cleared.")

def upsert_document(tx, filename: str, filepath: str, pages: List[str]):
    full_text = " ".join(pages)
    analysis = analyze_cv_guided(full_text)

    print(
        f"  -> AI: {filename} ({analysis.seniority}) - Found {len(analysis.assessed_skills)} skills, {len(analysis.companies)} companies.")

    # 1. Document & Metadata
    tx.run(
        """
        MERGE (d:Document {filename: $filename})
        ON CREATE SET d.path = $filepath, d.created = datetime()
        SET d.updated = datetime(),
            d.seniority = $seniority,
            d.summary = $summary,
            d.candidate_name = $candidate_name
        """,
        filename=filename,
        filepath=filepath,
        seniority=analysis.seniority,
        summary=analysis.summary,
        candidate_name=analysis.candidate_name
    )

    # 2. Pages
    if pages:
        tx.run(
            """
            MATCH (d:Document {filename: $filename})
            FOREACH (i IN range(0, size($pages)-1) |
                MERGE (p:Page {document: $filename, number: i+1})
                SET p.text = $pages[i]
                MERGE (d)-[:HAS_PAGE]->(p)
            )
            """,
            filename=filename,
            pages=pages
        )

    # 3. Skills
    skills_data = [{"name": s.name, "level": s.level} for s in analysis.assessed_skills]
    if skills_data:
        tx.run(
            """
            MATCH (d:Document {filename: $filename})
            FOREACH (skill IN $skills |
                MERGE (s:Skill {name: skill.name})
                SET s.id = skill.name
                MERGE (d)-[r:HAS_SKILL]->(s)
                SET r.level = skill.level
            )
            """,
            filename=filename,
            skills=skills_data
        )

    # 4. Companies
    if analysis.companies:
        tx.run(
            """
            MATCH (d:Document {filename: $filename})
            FOREACH (comp_name IN $companies |
                MERGE (c:Company {name: comp_name})
                SET c.id = comp_name
                MERGE (d)-[:WORKED_AT]->(c)
            )
            """,
            filename=filename,
            companies=analysis.companies
        )

    # 5. Projects
    if analysis.projects:
        tx.run(
            """
            MATCH (d:Document {filename: $filename})
            FOREACH (proj_name IN $projects |
                MERGE (p:Project {name: proj_name})
                SET p.id = proj_name
                MERGE (d)-[:WORKED_ON]->(p)
            )
            """,
            filename=filename,
            projects=analysis.projects
        )

    # 6. Universities
    if analysis.universities:
        tx.run(
            """
            MATCH (d:Document {filename: $filename})
            FOREACH (uni_name IN $universities |
                MERGE (u:University {name: uni_name})
                SET u.id = uni_name
                MERGE (d)-[:STUDIED_AT]->(u)
            )
            """,
            filename=filename,
            universities=analysis.universities
        )

    # 7. Certifications
    if analysis.certifications:
        tx.run(
            """
            MATCH (d:Document {filename: $filename})
            FOREACH (cert_name IN $certs |
                MERGE (c:Certification {name: cert_name})
                SET c.id = cert_name
                MERGE (d)-[:EARNED]->(c)
            )
            """,
            filename=filename,
            certs=analysis.certifications
        )

def save_graph_documents_manual(driver, graph_docs, include_source=True):
    with driver.session() as session:
        for doc in graph_docs:
            filename = doc.source.metadata.get("filename")

            def sanitize(text):
                return text.replace(" ", "_").replace("-", "_")

            for node in doc.nodes:
                label = sanitize(node.type)
                if not label:
                    label = "Unknown"

                props = {k: v for k, v in node.properties.items() if v is not None}

                query = f"MERGE (n:`{label}` {{id: $id}}) SET n += $props"
                session.run(query, id=node.id, props=props)

                if include_source and filename:
                    link_query = f"""
                    MATCH (d:Document {{filename: $filename}})
                    MATCH (n:`{label}` {{id: $id}})
                    MERGE (d)-[:MENTIONS]->(n)
                    """
                    session.run(link_query, filename=filename, id=node.id)

            for rel in doc.relationships:
                source = rel.source
                target = rel.target
                rel_type = sanitize(rel.type).upper()

                query = f"""
                MATCH (s:`{sanitize(source.type)}` {{id: $source_id}})
                MATCH (t:`{sanitize(target.type)}` {{id: $target_id}})
                MERGE (s)-[:{rel_type}]->(t)
                """
                session.run(query, source_id=source.id, target_id=target.id)

def process_file_with_graph_transformer(text: str, filename: str):
    print(f"  -> Graph: Extracting entities and relationships for {filename}...")
    try:
        doc = LangChainDocument(page_content=text, metadata={"filename": filename})
        graph_docs = llm_transformer.convert_to_graph_documents([doc])
        save_graph_documents_manual(driver, graph_docs, include_source=True)
        print(f"  -> Graph: Saved {len(graph_docs[0].nodes)} nodes and {len(graph_docs[0].relationships)} relationships.")
    except Exception as e:
        print(f"  -> Graph Extraction Error: {e}")

def check_file_exists(tx, filename):
    query = """
    MATCH (d:Document {filename: $filename})
    RETURN count(d) > 0 as exists
    """
    result = tx.run(query, filename=filename).single()
    return result["exists"] if result else False

# ------------------------------------------------------------------------------
# Główny Program
# Pętla przetwarzająca pliki w katalogu i zamykająca połączenie
# ------------------------------------------------------------------------------
def load_pdfs_to_neo4j(directory: Path = Path('.')):
    clear_database(driver)
    setup_database_constraints(driver)

    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        print("No PDF files found in", directory)
        return

    with driver.session() as session:
        for p in pdf_paths:
            print(f"Processing {p.name}...")

            try:
                pages = extract_text_from_pdf(p)
                full_text = " ".join(pages)

                session.execute_write(upsert_document, p.name, str(p.resolve()), pages)
                print(f"  -> Saved Metadata: {len(pages)} pages")

                process_file_with_graph_transformer(full_text, p.name)

            except Exception as e:
                print(f"Error processing {p.name}: {e}")

if __name__ == "__main__":
    load_pdfs_to_neo4j(Path('.'))
    driver.close()
