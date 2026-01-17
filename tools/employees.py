# tools/employees.py
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Konfiguracja i Stałe
# Inicjalizacja połączeń do Neo4j, OpenAI oraz ładowanie zmiennych środowiskowych
# ------------------------------------------------------------------------------
current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
project_root = current_dir.parent
env_path = project_root / '.env'
load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
    raise RuntimeError("Brak danych do Neo4j w .env")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

deploy_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
api_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
llm = AzureChatOpenAI(azure_deployment=deploy_name, api_version=api_ver)

# ------------------------------------------------------------------------------
# Modele Danych i Helpery
# Struktury Pydantic, ładowanie bazy umiejętności oraz konfiguracja ekstrakcji CV
# ------------------------------------------------------------------------------
def load_skills(filepath='skills-db.txt'):
    path = project_root / 'data' / filepath
    if not path.exists():
        path = project_root / filepath

    if not path.exists():
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

SKILLS_DB = load_skills('skills-db.txt')
known_skills_list = ", ".join(sorted(list(SKILLS_DB)))

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
        description="Lista nazw firm (pracodawców). Ujednolić nazwy (np. 'Google Inc.' -> 'Google')."
    )
    projects: List[str] = Field(description="Lista nazw kluczowych projektów.")
    universities: List[str] = Field(description="Lista nazw uczelni wyższych.")
    certifications: List[str] = Field(
        description="Lista nazw OFICJALNYCH certyfikatów branżowych (np. AWS Certified, ISTQB). Ignoruj kursy online."
    )

cv_extractor = llm.with_structured_output(CVAnalysis)

# ------------------------------------------------------------------------------
# Narzędzia - Zarządzanie Pracownikami
# Funkcje AI do obsługi kandydatów w bazie grafowej (CRUD)
# ------------------------------------------------------------------------------
@tool
def get_employees() -> str:
    """
    Pobiera listę pracowników (kandydatów) z bazy danych.
    Zwraca imię i nazwisko, seniority oraz podsumowanie.
    ZAWSZE używaj tego narzędzia jako pierwszego kroku, aby znaleźć dokładne 'candidate_name'
    przed próbą aktualizacji lub usunięcia pracownika.
    """
    query = """
    MATCH (d:Document)
    RETURN d.filename as filename, d.candidate_name as candidate_name, d.seniority as seniority, d.summary as summary
    ORDER BY d.candidate_name
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]

        if not records:
            return "Baza pracowników jest pusta."

        response_lines = []
        for r in records:
            c_name = r.get('candidate_name', 'Brak imienia')
            line = f"Imię: {c_name} | Plik: {r['filename']} | Poziom: {r['seniority']}"
            response_lines.append(line)
        return "\n".join(response_lines)

    except Exception as e:
        return f"Błąd bazy danych: {e}"

@tool
def delete_employee(candidate_name: str = None, filename: str = None) -> str:
    """
    Trwale USUWA pracownika z bazy Neo4j i odłącza wszystkie jego relacje.
    KRYTYCZNE: Operacja jest nieodwracalna. Wymaga wyraźnego POTWIERDZENIA od użytkownika.
    Przed usunięciem sprawdź, czy pracownik istnieje (użyj get_employees) i pokaż stan przed usunięciem.
    Wymagane podanie 'candidate_name' LUB 'filename'.
    """
    if not candidate_name and not filename:
        return "Błąd: Musisz podać 'candidate_name' lub 'filename' aby usunąć pracownika."

    check_query = """
    MATCH (d:Document)
    WHERE ($candidate_name IS NULL OR d.candidate_name = $candidate_name)
      AND ($filename IS NULL OR d.filename = $filename)
    RETURN d.candidate_name as name, d.filename as file
    """

    delete_query = """
    MATCH (d:Document)
    WHERE ($candidate_name IS NULL OR d.candidate_name = $candidate_name)
      AND ($filename IS NULL OR d.filename = $filename)
    DETACH DELETE d
    """

    try:
        with driver.session() as session:
            result = session.run(check_query, candidate_name=candidate_name, filename=filename)
            record = result.single()

            if not record:
                return "STAN PRZED: Nie znaleziono takiego pracownika.\nAKCJA: Brak.\nSTAN PO: Bez zmian."

            found_name = record['name']
            session.run(delete_query, candidate_name=candidate_name, filename=filename)

            return (f"STAN PRZED: Pracownik '{found_name}' istniał w bazie.\n"
                    f"AKCJA: Wykonano instrukcję DETACH DELETE.\n"
                    f"STAN PO: Pracownik i jego relacje zostały trwale usunięte.")

    except Exception as e:
        return f"Błąd podczas usuwania pracownika: {e}"

@tool
def create_employee(text: str, filename: Optional[str] = None, path: Optional[str] = None) -> str:
    """
    Analizuje tekst CV i tworzy NOWEGO pracownika w bazie.
    Narzędzie to wykonuje głęboką analizę tekstu przy użyciu LLM w celu ekstrakcji umiejętności, firm i certyfikatów.
    Używaj TYLKO dla nowych kandydatów. Wymaga POTWIERDZENIA, jeśli istnieje podejrzenie duplikatu.
    Umiejętności są walidowane względem globalnej listy SKILLS_DB.
    Nie wymyślaj danych, których nie ma w tekście.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
        Jesteś krytycznym i precyzyjnym analitykiem rekrutacyjnym.
        Twoim zadaniem jest ekstrakcja ustrukturyzowanych danych z surowego tekstu CV.

        ZASADY KRYTYCZNE:
        1. NIE DOPISUJ informacji, których nie ma w tekście (brak halucynacji).
        2. UMIEJĘTNOŚCI (Skills):
           - Porównuj znalezione technologie z poniższą listą dozwolonych (SKILLS_DB).
           - Jeśli kandydat zna 'ReactJS', a na liście jest 'React', zmapuj to.
           - Jeśli technologii nie ma na liście, POMIŃ JĄ.
           - Lista dozwolonych: {known_skills_list}
        3. CERTYFIKATY: Tylko oficjalne (Vendor Certified). Ignoruj kursy Udemy/Coursera bez egzaminu.
        4. JĘZYK: Podsumowanie (summary) napisz po polsku.
        """),
        ("human", "{cv_text}")
    ])

    safe_text = text[:15000]

    try:
        analysis = cv_extractor.invoke(prompt_template.invoke({"cv_text": safe_text}))
    except Exception as e:
        return f"Błąd modelu AI podczas analizy: {e}"

    skills_data = [
        {"name": s.name, "level": s.level}
        for s in analysis.assessed_skills
        if s.name in SKILLS_DB
    ]

    final_filename = filename or f"{analysis.candidate_name.replace(' ', '_')}_manual.pdf"
    final_path = path or ""

    try:
        with driver.session() as session:
            check = session.run("MATCH (d:Document {filename: $f}) RETURN d", f=final_filename)
            if check.single():
                return f"STAN PRZED: Pracownik z plikiem '{final_filename}' już istnieje.\nAKCJA: Anulowano.\nSTAN PO: Bez zmian. Użyj update_employee."

            session.run(
                """
                MERGE (d:Document {filename: $filename})
                ON CREATE SET d.created = datetime()
                SET d.candidate_name = $candidate_name,
                    d.seniority = $seniority,
                    d.summary = $summary,
                    d.path = $path,
                    d.updated = datetime()
                """,
                filename=final_filename,
                candidate_name=analysis.candidate_name,
                seniority=analysis.seniority,
                summary=analysis.summary,
                path=final_path
            )

            if skills_data:
                session.run(
                    """
                    MATCH (d:Document {filename: $filename})
                    FOREACH (skill IN $skills |
                        MERGE (s:Skill {name: skill.name})
                        MERGE (d)-[r:HAS_SKILL]->(s)
                        SET r.level = skill.level
                    )
                    """,
                    filename=final_filename,
                    skills=skills_data
                )

            if analysis.companies:
                session.run(
                    "MATCH (d:Document {filename: $f}) FOREACH (c IN $l | MERGE (n:Company {name:c}) MERGE (d)-[:WORKED_AT]->(n))",
                    f=final_filename, l=analysis.companies)

            if analysis.certifications:
                session.run(
                    "MATCH (d:Document {filename: $f}) FOREACH (c IN $l | MERGE (n:Certification {name:c}) MERGE (d)-[:EARNED]->(n))",
                    f=final_filename, l=analysis.certifications)

            if analysis.projects:
                session.run(
                    "MATCH (d:Document {filename: $f}) FOREACH (p IN $l | MERGE (n:Project {name:p}) MERGE (d)-[:WORKED_ON]->(n))",
                    f=final_filename, l=analysis.projects)

            if analysis.universities:
                session.run(
                    "MATCH (d:Document {filename: $f}) FOREACH (u IN $l | MERGE (n:University {name:u}) MERGE (d)-[:STUDIED_AT]->(n))",
                    f=final_filename, l=analysis.universities)

        return (f"STAN PRZED: Brak wpisu dla '{final_filename}'.\n"
                f"AKCJA: Utworzono profil kandydata: {analysis.candidate_name}.\n"
                f"STAN PO: Zapisano {len(skills_data)} umiejętności, {len(analysis.companies)} firm. Poziom: {analysis.seniority}.")

    except Exception as e:
        return f"Błąd bazy danych podczas zapisu: {e}"

@tool
def update_employee(
        current_candidate_name: str,
        new_candidate_name: Optional[str] = None,
        new_seniority: Optional[str] = None,
        new_summary: Optional[str] = None,
        new_email: Optional[str] = None,
        new_phone: Optional[str] = None,
        skills_to_add: Optional[List[str]] = None,
        skills_to_remove: Optional[List[str]] = None
) -> str:
    """
    Aktualizuje dane ISNIEJĄCEGO pracownika.
    Wymaga POTWIERDZENIA użytkownika przed wprowadzeniem zmian.
    Argument 'skills_to_add' przyjmuje listę stringów w formacie "Nazwa:Poziom" (np. "Python:Expert").
    Umiejętności są walidowane przez SKILLS_DB.
    Zwraca raport ze stanem przed i po zmianie.
    """
    processed_adds = []
    if skills_to_add:
        for item in skills_to_add:
            parts = item.split(':', 1)
            s_name = parts[0].strip()
            s_level = parts[1].strip() if len(parts) > 1 else "Intermediate"
            if s_name in SKILLS_DB:
                processed_adds.append({"name": s_name, "level": s_level})

    safe_adds = processed_adds if processed_adds else []
    safe_removes = skills_to_remove if skills_to_remove else []

    fetch_query = """
    MATCH (d:Document {candidate_name: $name})
    OPTIONAL MATCH (d)-[r:HAS_SKILL]->(s:Skill)
    RETURN d.seniority as seniority, d.email as email, d.phone as phone, collect(s.name) as skills
    """

    update_query = """
    MATCH (d:Document {candidate_name: $current_name})

    // Scalar Updates
    SET d.candidate_name = COALESCE($new_name, d.candidate_name),
        d.seniority = COALESCE($new_seniority, d.seniority),
        d.summary = COALESCE($new_summary, d.summary),
        d.email = COALESCE($new_email, d.email),
        d.phone = COALESCE($new_phone, d.phone),
        d.updated = datetime()

    WITH d

    // Remove Skills
    OPTIONAL MATCH (d)-[r_rem:HAS_SKILL]->(s_rem:Skill)
    WHERE s_rem.name IN $removes
    DELETE r_rem

    WITH d

    // Add/Update Skills
    FOREACH (skill IN $adds |
        MERGE (s:Skill {name: skill.name})
        MERGE (d)-[r:HAS_SKILL]->(s)
        SET r.level = skill.level
    )

    // Return Post-State
    WITH d
    MATCH (d)-[r:HAS_SKILL]->(s:Skill)
    RETURN d.candidate_name as name, d.seniority as seniority, d.email as email, collect(s.name + ' (' + r.level + ')') as current_skills
    """

    try:
        with driver.session() as session:
            before_res = session.run(fetch_query, name=current_candidate_name)
            before_rec = before_res.single()

            if not before_rec:
                return f"Błąd: Nie znaleziono pracownika o nazwisku '{current_candidate_name}'."

            before_desc = (f"Seniority: {before_rec['seniority']}, "
                           f"Email: {before_rec.get('email', 'brak')}, "
                           f"Liczba skilli: {len(before_rec['skills'])}")

            update_res = session.run(
                update_query,
                current_name=current_candidate_name,
                new_name=new_candidate_name,
                new_seniority=new_seniority,
                new_summary=new_summary,
                new_email=new_email,
                new_phone=new_phone,
                removes=safe_removes,
                adds=safe_adds
            )
            after_rec = update_res.single()

            if after_rec:
                skills_str = ", ".join(after_rec['current_skills'][:5]) + (
                    "..." if len(after_rec['current_skills']) > 5 else "")
                return (f"STAN PRZED: {before_desc}\n"
                        f"AKCJA: Zaktualizowano pola podstawowe i umiejętności.\n"
                        f"STAN PO: Imię: {after_rec['name']}, Seniority: {after_rec['seniority']}, "
                        f"Email: {after_rec.get('email', 'brak')}, Skills: {skills_str}")
            else:
                return "STAN PO: Dane zaktualizowane, ale pracownik nie posiada teraz żadnych skilli."

    except Exception as e:
        return f"Błąd podczas aktualizacji: {e}"
