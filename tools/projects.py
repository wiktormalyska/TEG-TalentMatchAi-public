# tools/projects.py
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
from langchain_core.tools import tool

# ------------------------------------------------------------------------------
# Konfiguracja i Stałe
# Definicje ścieżek do plików bazy danych
# ------------------------------------------------------------------------------
PROJECTS_DB_PATH = Path(__file__).resolve().parents[1] / "projects.json"
ASSIGNMENTS_DB_PATH = Path(__file__).resolve().parents[1] / "assignments.json"

# ------------------------------------------------------------------------------
# Modele Danych
# Struktury danych dla projektów i przypisań
# ------------------------------------------------------------------------------
@dataclass
class ProjectDefinition:
    id: int
    project_name: str
    description: str
    start_date: str
    end_date: str
    status: str = "Oczekiwanie na oferty"
    notes: str = ""
    type: str = "project_definition"

@dataclass
class ProjectAssignment:
    id: int
    employee_name: str
    project_name: str
    start_date: str
    end_date: str
    fte: float
    type: str = "assignment"

# ------------------------------------------------------------------------------
# Operacje Wejścia/Wyjścia
# Funkcje pomocnicze do odczytu i zapisu JSON
# ------------------------------------------------------------------------------
def _save_projects(data: List[ProjectDefinition]):
    PROJECTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized_data = [asdict(item) for item in data]
    with open(PROJECTS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, ensure_ascii=False)

def _load_projects() -> List[ProjectDefinition]:
    if not PROJECTS_DB_PATH.exists():
        _save_projects([])
        return []

    try:
        with open(PROJECTS_DB_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            return []

        items = []
        for item in raw_data:
            if not isinstance(item, dict): continue
            items.append(ProjectDefinition(
                id=item.get("id", 0),
                project_name=item.get("project_name", ""),
                description=item.get("description", ""),
                start_date=item.get("start_date", ""),
                end_date=item.get("end_date", ""),
                status=item.get("status", "Nieznany"),
                notes=item.get("notes", "")
            ))
        return items
    except (json.JSONDecodeError, IOError):
        return []

def _save_assignments(data: List[ProjectAssignment]):
    ASSIGNMENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized_data = [asdict(item) for item in data]
    with open(ASSIGNMENTS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, ensure_ascii=False)

def _load_assignments() -> List[ProjectAssignment]:
    if not ASSIGNMENTS_DB_PATH.exists():
        _save_assignments([])
        return []

    try:
        with open(ASSIGNMENTS_DB_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            return []

        items = []
        for item in raw_data:
            if not isinstance(item, dict): continue
            items.append(ProjectAssignment(
                id=item.get("id", 0),
                employee_name=item.get("employee_name", ""),
                project_name=item.get("project_name", ""),
                start_date=item.get("start_date", ""),
                end_date=item.get("end_date", ""),
                fte=float(item.get("fte", 1.0))
            ))
        return items
    except (json.JSONDecodeError, IOError):
        return []

# ------------------------------------------------------------------------------
# Narzędzia - Projekty
# Funkcje AI do zarządzania definicjami projektów
# ------------------------------------------------------------------------------
@tool
def create_project(project_name: str, description: str, start_date: str, end_date: str, status: str = "Oczekiwanie na oferty", notes: str = "") -> str:
    """Tworzy wpis o nowym projekcie w bazie danych (definicja projektu)."""
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Błąd: Format daty musi być YYYY-MM-DD."

    data = _load_projects()

    for item in data:
        if item.project_name.lower() == project_name.lower():
            return f"Błąd: Projekt o nazwie '{project_name}' już istnieje."

    new_project = ProjectDefinition(
        id=len(data) + 1,
        project_name=project_name,
        description=description,
        start_date=start_date,
        end_date=end_date,
        status=status,
        notes=notes
    )

    data.append(new_project)
    _save_projects(data)
    return f"Pomyślnie utworzono projekt: {project_name} (Status: {status})"

@tool
def list_projects() -> str:
    """Zwraca listę wszystkich zdefiniowanych projektów."""
    data = _load_projects()
    projects = [asdict(item) for item in data]

    if not projects:
        return "Brak zdefiniowanych projektów w bazie."

    return json.dumps(projects, indent=2, ensure_ascii=False)

@tool
def update_project(project_name: str, status: Optional[str] = None, description: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, notes: Optional[str] = None) -> str:
    """Aktualizuje dane istniejącego projektu."""
    data = _load_projects()
    found = False

    if start_date:
        try: datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError: return "Błąd: start_date musi być YYYY-MM-DD."
    if end_date:
        try: datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError: return "Błąd: end_date musi być YYYY-MM-DD."

    for item in data:
        if item.project_name.lower() == project_name.lower():
            if status: item.status = status
            if description: item.description = description
            if start_date: item.start_date = start_date
            if end_date: item.end_date = end_date
            if notes: item.notes = notes
            found = True
            break

    if found:
        _save_projects(data)
        return f"Zaktualizowano projekt: {project_name}"
    return f"Nie znaleziono projektu o nazwie: {project_name}"

@tool
def delete_project(project_name: str) -> str:
    """Usuwa definicję projektu z bazy."""
    data = _load_projects()
    initial_len = len(data)

    new_data = [d for d in data if d.project_name.lower() != project_name.lower()]

    if len(new_data) == initial_len:
        return f"Nie znaleziono projektu: {project_name}"

    _save_projects(new_data)
    return f"Usunięto projekt: {project_name}"

# ------------------------------------------------------------------------------
# Narzędzia - Alokacja Zasobów
# Funkcje AI do przypisywania pracowników do projektów
# ------------------------------------------------------------------------------
@tool
def get_project_assignments(employee_name: Optional[str] = None) -> str:
    """Pobiera listę przypisań projektowych (z pliku assignments.json)."""
    data = _load_assignments()
    result = []

    for item in data:
        if employee_name and employee_name.lower() not in item.employee_name.lower():
            continue
        result.append(asdict(item))

    if not result:
        return "Brak przypisań w bazie."

    return json.dumps(result, indent=2, ensure_ascii=False)

@tool
def add_project_assignment(employee_name: str, project_name: str, start_date: str, end_date: str, fte: float) -> str:
    """Dodaje przypisanie pracownika do projektu."""
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Błąd: Format daty musi być YYYY-MM-DD."

    data = _load_assignments()

    new_assignment = ProjectAssignment(
        id=len(data) + 1,
        employee_name=employee_name,
        project_name=project_name,
        start_date=start_date,
        end_date=end_date,
        fte=float(fte)
    )

    data.append(new_assignment)
    _save_assignments(data)
    return f"Pomyślnie dodano przypisanie: {employee_name} do {project_name} ({fte} FTE)."

@tool
def delete_project_assignment(employee_name: str, project_name: str) -> str:
    """Usuwa przypisanie pracownika do konkretnego projektu."""
    data = _load_assignments()
    initial_len = len(data)

    new_data = []
    for item in data:
        if item.employee_name.lower() == employee_name.lower() and item.project_name.lower() == project_name.lower():
            continue
        new_data.append(item)

    if len(new_data) == initial_len:
        return "Nie znaleziono takiego przypisania."

    _save_assignments(new_data)
    return "Przypisanie usunięte."

@tool
def check_availability(check_date: str) -> str:
    """Sprawdza dostępność i obłożenie pracowników na wskazaną datę."""
    try:
        target_dt = datetime.strptime(check_date, "%Y-%m-%d")
    except ValueError:
        return "Błąd: Format daty musi być YYYY-MM-DD."

    data = _load_assignments()
    usage_map = {}

    for item in data:
        try:
            start_dt = datetime.strptime(item.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(item.end_date, "%Y-%m-%d")

            if start_dt <= target_dt <= end_dt:
                name = item.employee_name
                usage_map[name] = usage_map.get(name, 0.0) + item.fte
        except ValueError:
            continue

    report = [f"--- Raport Dostępności na dzień {check_date} ---"]
    found_anyone = False

    for name, usage in usage_map.items():
        found_anyone = True
        bench = 1.0 - usage
        status = "Przeciążony" if bench < 0 else ("Dostępny" if bench > 0 else "W pełni obłożony")
        report.append(f"- {name}: Zajętość {usage:.2f} FTE | Wolne moce (Bench): {bench:.2f} FTE | [{status}]")

    if not found_anyone:
        return f"Na dzień {check_date} nie znaleziono aktywnych przypisań."

    return "\n".join(report)
