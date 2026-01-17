# tools/skills.py
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from neo4j import GraphDatabase

# --- Configuration & Database Connection ---
# Sets up environment variables and initializes the Neo4j database driver.

current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
project_root = current_dir.parent
env_path = project_root / '.env'

load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
    raise RuntimeError("Missing Neo4j credentials in .env")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# --- Helper Functions ---
# Internal utilities for file handling logic.

def get_skills_file_path() -> Path:
    path = project_root / 'data' / 'skills-db.txt'
    if not path.exists():
        path = project_root / 'skills-db.txt'
    return path


# --- Data Retrieval Tools ---
# Tools exposed to AI for fetching system data.

@tool
def get_skills() -> str:
    """
    Pobiera listę znanych/dozwolonych umiejętności (Skills) z systemu.
    Użyj tego, aby sprawdzić pisownię lub dostępność skilla przed dodaniem/usunięciem.
    """
    skills_file = get_skills_file_path()
    if not skills_file.exists():
        return "Baza umiejętności jest pusta (brak pliku)."

    with open(skills_file, 'r', encoding='utf-8') as f:
        skills = [line.strip() for line in f if line.strip()]

    return "Aktualna lista Skillów:\n" + ", ".join(sorted(skills))


# --- Data Modification Tools ---
# Tools exposed to AI for creating, updating, or deleting system data.

@tool
def create_skills(skill_name: str) -> str:
    """
    Dodaje NOWĄ umiejętność do globalnej bazy systemu (Neo4j + plik txt).
    Użyj tego, gdy kandydat posiada skill, którego nie ma jeszcze na liście.
    Argument 'skill_name' musi być ścisłym ciągiem znaków (np. 'Python', a nie 'Programowanie w Pythonie').
    Zwraca raport ze stanem przed i po operacji.
    """
    skills_file = get_skills_file_path()

    if not skills_file.parent.exists():
        skills_file.parent.mkdir(parents=True, exist_ok=True)

    current_skills = set()
    if skills_file.exists():
        with open(skills_file, 'r', encoding='utf-8') as f:
            current_skills = {line.strip() for line in f if line.strip()}

    if skill_name in current_skills:
        return f"STAN PRZED: Skill '{skill_name}' już istnieje.\nAKCJA: Brak.\nSTAN PO: Bez zmian."

    try:
        with driver.session() as session:
            session.run("MERGE (s:Skill {name: $name})", name=skill_name)
    except Exception as e:
        return f"Błąd Neo4j podczas dodawania: {e}"

    try:
        with open(skills_file, "a", encoding="utf-8") as f:
            f.write(f"\n{skill_name}")

        return (f"STAN PRZED: Skilla '{skill_name}' brak w bazie.\n"
                f"AKCJA: Dodano skill do Neo4j i pliku indeksu.\n"
                f"STAN PO: Skill '{skill_name}' jest teraz dostępny w systemie.")

    except Exception as e:
        return f"Błąd pliku podczas zapisu: {e}"


@tool
def delete_skill(skill_name: str) -> str:
    """
    Trwale USUWA umiejętność z bazy Neo4j i pliku lokalnego.
    KRYTYCZNE: Użyj tylko po wyraźnym i ostatecznym POTWIERDZENIU przez użytkownika.
    Usunięcie skilla spowoduje zniknięcie relacji u wszystkich pracowników posiadających tę umiejętność.
    Przed wywołaniem sprawdź za pomocą 'get_skills' czy skill faktycznie istnieje.
    """
    skills_file = get_skills_file_path()

    exists_before = False
    current_skills = []
    if skills_file.exists():
        with open(skills_file, 'r', encoding='utf-8') as f:
            current_skills = [line.strip() for line in f if line.strip()]
        if skill_name in current_skills:
            exists_before = True

    if not exists_before:
        return f"STAN PRZED: Skill '{skill_name}' nie istnieje.\nAKCJA: Brak.\nSTAN PO: Bez zmian."

    try:
        with driver.session() as session:
            session.run("MATCH (s:Skill {name: $name}) DETACH DELETE s", name=skill_name)
    except Exception as e:
        return f"Błąd Neo4j podczas usuwania: {e}"

    try:
        new_skills = [s for s in current_skills if s != skill_name]
        with open(skills_file, "w", encoding="utf-8") as f:
            f.write("\n".join(new_skills))

        return (f"STAN PRZED: Skill '{skill_name}' istniał w bazie.\n"
                f"AKCJA: Wykonano DETACH DELETE w Neo4j i usunięto z pliku.\n"
                f"STAN PO: Skill został trwale usunięty.")
    except Exception as e:
        return f"Błąd pliku podczas usuwania: {e}"


@tool
def update_skill(old_name: str, new_name: str) -> str:
    """
    Zmienia nazwę ISTNIEJĄCEJ umiejętności w bazie i pliku (np. poprawa literówki).
    Wymaga POTWIERDZENIA użytkownika - zmiana wpłynie na historię danych.
    1. Upewnij się, że 'old_name' istnieje w 'get_skills'.
    2. 'new_name' to nowa poprawna nazwa.
    Zwraca raport zmian.
    """
    skills_file = get_skills_file_path()

    if skills_file.exists():
        with open(skills_file, "r", encoding="utf-8") as f:
            skills = [line.strip() for line in f if line.strip()]

        if old_name not in skills:
            return f"STAN PRZED: Skill '{old_name}' nie istnieje w pliku indeksu.\nAKCJA: Przerwano aktualizację."

    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (s:Skill {name: $old})
                SET s.name = $new
                RETURN s
                """,
                old=old_name, new=new_name
            )
            if not result.single():
                return f"STAN PRZED: Skill '{old_name}' nie znaleziony w Neo4j.\nAKCJA: Zaktualizowano tylko plik (jeśli istniał)."
    except Exception as e:
        return f"Błąd Neo4j podczas aktualizacji: {e}"

    try:
        skills = [new_name if s == old_name else s for s in skills]
        skills = sorted(list(set(skills)))

        with open(skills_file, "w", encoding="utf-8") as f:
            f.write("\n".join(skills))

        return (f"STAN PRZED: Nazwa skilla brzmiała '{old_name}'.\n"
                f"AKCJA: Aktualizacja nazwy na '{new_name}'.\n"
                f"STAN PO: Skill widnieje w systemie jako '{new_name}'.")

    except Exception as e:
        return f"Błąd pliku podczas aktualizacji: {e}"
