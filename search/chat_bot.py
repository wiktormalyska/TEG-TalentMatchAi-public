# search/chat_bot.py (app.py)
import os
import sys
import io
import json
import logging
from pathlib import Path
from typing import TypedDict, Annotated, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import docx
except ImportError:
    docx = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.employees import get_employees, create_employee, delete_employee, update_employee
from tools.skills import get_skills, create_skills, delete_skill, update_skill
from tools.projects import get_project_assignments, add_project_assignment, delete_project_assignment, check_availability, delete_project, create_project, update_project, list_projects

# ------------------------------------------------------------------------------
# Konfiguracja Środowiska
# Inicjalizacja logowania, zmiennych środowiskowych i modelu OpenAI
# ------------------------------------------------------------------------------
current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
project_root = current_dir
env_path = project_root / ".env"

load_dotenv(env_path)

deploy_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5-nano"
api_ver = os.getenv("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

system_logger = logging.getLogger("system_streamlit")
system_logger.info(f"Using Azure Deployment: {deploy_name}, API Version: {api_ver}")

llm = AzureChatOpenAI(
    azure_deployment=deploy_name,
    api_version=api_ver,
)

# ------------------------------------------------------------------------------
# Logika Biznesowa - Scoring
# Funkcja oceny dopasowania kandydata do wymagań RFP
# ------------------------------------------------------------------------------
def match_rfp_scoring(
        candidate_skills: List[str],
        must_have_skills: List[str],
        nice_to_have_skills: List[str],
        availability_fte: float
) -> str:
    """
    Silnik oceny (Scoring Engine) dla dopasowania kandydata do RFP.
    Oblicza wynik punktowy (0-100%) biorąc pod uwagę wagi:
    - Must-Have: 50%
    - Nice-To-Have: 20%
    - Dostępność (Bench/Availability): 30%

    Zwraca sformatowany string z wynikiem i uzasadnieniem.
    """
    cand_skills_norm = {s.lower().strip() for s in candidate_skills}
    must_norm = {s.lower().strip() for s in must_have_skills if s.strip()}
    nice_norm = {s.lower().strip() for s in nice_to_have_skills if s.strip()}

    if not must_norm:
        must_score = 1.0
    else:
        must_matches = must_norm.intersection(cand_skills_norm)
        must_score = len(must_matches) / len(must_norm)

    if not nice_norm:
        nice_score = 0.0
    else:
        nice_matches = nice_norm.intersection(cand_skills_norm)
        nice_score = len(nice_matches) / len(nice_norm)

    avail_score = max(0.0, min(1.0, availability_fte))

    w_must = 0.5
    w_nice = 0.2
    w_avail = 0.3

    final_score = (must_score * w_must) + (nice_score * w_nice) + (avail_score * w_avail)
    final_percent = round(final_score * 100, 1)

    return json.dumps({
        "total_score": final_percent,
        "details": {
            "must_have_match": f"{len(must_norm.intersection(cand_skills_norm))}/{len(must_norm)}",
            "nice_to_have_match": f"{len(nice_norm.intersection(cand_skills_norm))}/{len(nice_norm)}",
            "availability_fte": availability_fte
        },
        "verdict": "Dobry kandydat" if final_percent > 70 else "Wymaga weryfikacji"
    })

# ------------------------------------------------------------------------------
# Konfiguracja AI i LangGraph
# Definicja narzędzi, grafu stanów oraz promptu systemowego
# ------------------------------------------------------------------------------
tools = [
    get_employees, create_employee, delete_employee, update_employee,
    get_skills, create_skills, delete_skill, update_skill,
    get_project_assignments, add_project_assignment, delete_project_assignment, check_availability,
    match_rfp_scoring, create_project, update_project, list_projects, delete_project
]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

system_prompt = """JESTEŚ WYSOKIEJ KLASY SYSTEMEM HR MATCHING ENGINE I ZARZĄDZANIA BAZĄ DANYCH.
NIE JESTEŚ ASYSTENTEM "CHATBOTEM" DO POGAWĘDEK, LECZ PRECYZYJNYM NARZĘDZIEM OPERACYJNYM.

DYREKTYWY KRYTYCZNE (BEZWZGLĘDNE):
1. ZAKAZ HALUCYNACJI: Nie wolno Ci zmyślać pracowników, umiejętności, dostępności ani projektów.
2. JEŚLI CZEGOŚ NIE MA W BAZIE: Informuj wprost: "Brak danych w systemie". Nie generuj przykładowych danych.
3. STRICT TOOL USAGE: Każdą informację musisz pobrać przez odpowiednie narzędzie. Nie wolno Ci zgadywać.

PROTOKOŁY OPERACYJNE:

PRZYPADEK A: ANALIZA CV (RESUME)
1. EKSTRAKCJA: Wyodrębnij z tekstu Imię, Nazwisko, Listę Skillów, Obecną Rolę.
2. AKCJA: Użyj `create_employee` (dla pracownika) oraz `create_skills` (dla umiejętności technicznych).
3. RAPORT: "Zarejestrowano kandydata: [Imię Nazwisko] | Rola: [Wykryta Rola] | Skille: [Lista]". Bądź zwięzły.

PRZYPADEK B: MATCHING POD RFP (ZAPYTANIE OFERTOWE)
Musisz wykonać sekwencję logiczną w dokładnie tej kolejności:
KROK 1 (Analiza): Wyodrębnij 'Must-Have Skills', 'Nice-To-Have Skills' oraz 'Wymagane FTE/Start'.
KROK 2 (Szukanie): Uruchom `get_employees` filtrując po kluczowych 'Must-Have skills'. Jeśli lista pusta -> STOP i poinformuj.
KROK 3 (Dostępność): Dla KAŻDEGO znalezionego kandydata wykonaj `check_availability`. To krytyczne dla wyniku.
KROK 4 (Scoring): Wywołaj `match_rfp_scoring` dla każdego kandydata, podając parametry z RFP oraz wynik dostępności z kroku 3.
KROK 5 (Prezentacja): Wygeneruj tabelę Markdown.

FORMAT WYJŚCIOWY (STRICT MARKDOWN TABLE):
| Kandydat | Wynik Dopasowania | Bench (FTE) | Must-Have (x/y) | Decyzja |
|----------|-------------------|-------------|-----------------|---------|
| Jan Kowalski | 85.0% | 1.0 (Wolny) | 4/5 | Rekomendowany |

STYL KOMUNIKACJI:
- Profesjonalny, bezosobowy, "surowy" (Data-Driven).
- Żadnych zbędnych przymiotników ("Wspaniały", "Niesamowity").
- Tylko fakty wynikające z wywołania narzędzi.

Pamiętaj: Jeśli narzędzie `check_availability` zwróci 'assigned', to `availability_fte` do scoringu wynosi 0.0. Jeśli brak przypisań, to 1.0."""

# ------------------------------------------------------------------------------
# Funkcje Pomocnicze
# Obsługa plików przesyłanych przez użytkownika
# ------------------------------------------------------------------------------
def read_uploaded_file(uploaded_file):
    """Odczytuje tekst z przesłanego pliku (PDF, DOCX, TXT)."""
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            if PyPDF2:
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            else:
                return "Błąd: Brak biblioteki PyPDF2."
        elif uploaded_file.name.endswith('.docx'):
            if docx:
                doc = docx.Document(uploaded_file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                return "Błąd: Brak biblioteki python-docx."
        else:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
    except Exception as e:
        return f"Błąd podczas odczytu pliku: {e}"

    return text

# ------------------------------------------------------------------------------
# Interfejs Użytkownika (Streamlit)
# Konfiguracja strony, zarządzanie sesją i sidebar
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="HR TalentMatch AI",
    page_icon="🧑‍💻",
    layout="centered",
)

st.title("🧑‍💻 HR TalentMatch AI")
st.caption(
    "Asystent HR: Baza pracowników, Upload CV/RFP, Matching Engine & Bench.")

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = "thread-1"
    st.session_state.threads["thread-1"] = {
        "name": "Nowy wątek",
        "messages": [SystemMessage(content=system_prompt)],
        "auto_named": False,
    }

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.sidebar:
    st.header("Wątki")

    if st.button("➕ Nowy wątek", use_container_width=True):
        new_id = f"thread-{len(st.session_state.threads) + 1}"
        st.session_state.threads[new_id] = {
            "name": "Nowy wątek",
            "messages": [SystemMessage(content=system_prompt)],
            "auto_named": False,
        }
        st.session_state.current_thread_id = new_id
        st.rerun()

    st.markdown("---")

    for tid, data in list(st.session_state.threads.items()):
        is_current = tid == st.session_state.current_thread_id
        display_name = data["name"] or "Bez nazwy"

        row_clicked = st.button(
            ("▶ " if is_current else "☰ ") + display_name,
            key=f"row-{tid}",
            use_container_width=True,
        )
        if row_clicked and not is_current:
            st.session_state.current_thread_id = tid
            st.rerun()

        with st.expander("Opcje", expanded=False):
            new_name = st.text_input("Nazwa wątku", value=data["name"], key=f"name-{tid}")
            if new_name != data["name"]:
                st.session_state.threads[tid]["name"] = new_name

            cols_opt = st.columns([0.5, 0.5])
            with cols_opt[0]:
                if st.button("Wybierz", key=f"set-{tid}"):
                    st.session_state.current_thread_id = tid
                    st.rerun()
            with cols_opt[1]:
                if st.button("Usuń", key=f"delete-{tid}"):
                    if len(st.session_state.threads) > 1:
                        del st.session_state.threads[tid]
                        if st.session_state.current_thread_id == tid:
                            st.session_state.current_thread_id = next(iter(st.session_state.threads.keys()))
                    else:
                        st.warning("Nie można usunąć jedynego wątku.")
                    st.rerun()

# ------------------------------------------------------------------------------
# Główne Okno - Czat i Obsługa Plików
# Logika interakcji, przesyłania plików i odpowiedzi AI
# ------------------------------------------------------------------------------
current_thread = st.session_state.current_thread_id
thread_data = st.session_state.threads[current_thread]
messages = thread_data["messages"]

for msg in messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    content_to_show = msg.content
    if role == "user" and "TREŚĆ PLIKU:" in msg.content:
        content_to_show = "📄 *[Użytkownik przesłał plik do analizy (CV lub RFP)]*"

    with st.chat_message(role):
        st.markdown(content_to_show)

with st.expander("📎 Wgraj plik (CV lub RFP)", expanded=False):
    st.caption("System automatycznie rozpozna czy to CV do bazy, czy RFP do dopasowania.")
    col_file, col_btn = st.columns([4, 1], gap="small")
    with col_file:
        uploaded_file = st.file_uploader(
            "Wybierz plik",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
    with col_btn:
        st.write("")
        analyze_clicked = st.button("Prześlij", use_container_width=True)

    if uploaded_file and analyze_clicked:
        with st.spinner("Analiza dokumentu i rozpoznawanie typu..."):
            file_text = read_uploaded_file(uploaded_file)

            if file_text and not file_text.startswith("Błąd"):
                prompt_content = (
                    f"TREŚĆ PLIKU:\n{file_text[:30000]}\n\n"
                    f"Wykonaj procedurę zgodnie z System Prompt (CV=Import, RFP=Matching)."
                )

                st.session_state.threads[current_thread]["messages"].append(
                    HumanMessage(content=prompt_content)
                )

                non_system = [m for m in st.session_state.threads[current_thread]["messages"] if
                              not isinstance(m, SystemMessage)]
                if len(non_system) == 1 and not thread_data.get("auto_named", False):
                    fname = uploaded_file.name
                    if len(fname) > 20:
                        fname = fname[:20] + "..."
                    st.session_state.threads[current_thread]["name"] = f"Plik: {fname}"
                    st.session_state.threads[current_thread]["auto_named"] = True

                st.session_state.uploader_key += 1
                st.rerun()
            else:
                st.error(f"Nie udało się odczytać pliku. {file_text}")

prompt = st.chat_input("Zadaj pytanie HR botowi (np. znajdź Java Dev)...")

if prompt:
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    if not non_system_msgs and not thread_data.get("auto_named", False):
        auto_name = prompt.strip()
        if len(auto_name) > 40:
            auto_name = auto_name[:40] + "..."
        st.session_state.threads[current_thread]["name"] = auto_name or "Nowy wątek"
        st.session_state.threads[current_thread]["auto_named"] = True

    messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

if messages and isinstance(messages[-1], HumanMessage):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        config = {"configurable": {"thread_id": current_thread}}

        events = graph.stream(
            {"messages": messages},
            config,
            stream_mode="values",
        )

        last_ai_msg = None
        for event in events:
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]

                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_name = tc.get('name')
                        tool_args = tc.get('args')
                        print(f"\033[93m🛠️ [DEBUG] AI uses tool: {tool_name} | Args: {tool_args}\033[0m")

                if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                    last_ai_msg = last_msg
                    full_response = last_msg.content
                    message_placeholder.markdown(full_response)

        if last_ai_msg:
            if messages[-1] != last_ai_msg:
                messages.append(AIMessage(content=last_ai_msg.content))
            st.session_state.threads[current_thread]["messages"] = messages
            st.rerun()


# to run app:
# streamlit run chat_bot.py