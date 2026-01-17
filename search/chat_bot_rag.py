# search/chat_bot_rag.py
import os
import sys
import io
import json
import logging
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Using only langchain_core and openai prevents ImportError from missing 'langchain' package
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI

# Optional imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import docx
except ImportError:
    docx = None

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.employees import get_employees, create_employee, delete_employee, update_employee
from tools.skills import get_skills, create_skills, delete_skill, update_skill
from tools.projects import get_project_assignments, add_project_assignment, delete_project_assignment, check_availability, delete_project, create_project, update_project, list_projects

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
project_root = current_dir
env_path = project_root / ".env"

load_dotenv(env_path)

deploy_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5-nano"
api_ver = os.getenv("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

llm = AzureChatOpenAI(
    azure_deployment=deploy_name,
    api_version=api_ver,
    streaming=True
)

# ------------------------------------------------------------------------------
# Business Logic (Scoring)
# ------------------------------------------------------------------------------
def match_rfp_scoring(
        candidate_skills: List[str],
        must_have_skills: List[str],
        nice_to_have_skills: List[str],
        availability_fte: float
) -> str:
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

    final_score = (must_score * 0.5) + (nice_score * 0.2) + (avail_score * 0.3)
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
# Agent Setup (Manual Loop - No AgentExecutor)
# ------------------------------------------------------------------------------
tools = [
    get_employees, create_employee, delete_employee, update_employee,
    get_skills, create_skills, delete_skill, update_skill,
    get_project_assignments, add_project_assignment, delete_project_assignment, check_availability,
    match_rfp_scoring, create_project, update_project, list_projects, delete_project
]

# FIX: Use .name for StructuredTool objects, fallback to __name__ for functions
tool_map = {t.name if hasattr(t, "name") else t.__name__: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

system_prompt = """JESTEŚ WYSOKIEJ KLASY SYSTEMEM HR MATCHING ENGINE I ZARZĄDZANIA BAZĄ DANYCH.
NIE JESTEŚ ASYSTENTEM "CHATBOTEM" DO POGAWĘDEK, LECZ PRECYZYJNYM NARZĘDZIEM OPERACYJNYM.

DYREKTYWY KRYTYCZNE (BEZWZGLĘDNE):
1. ZAKAZ HALUCYNACJI: Nie wolno Ci zmyślać pracowników, umiejętności, dostępności ani projektów.
2. JEŚLI CZEGOŚ NIE MA W BAZIE: Informuj wprost: "Brak danych w systemie". Nie generuj przykładowych danych.
3. STRICT TOOL USAGE: Każdą informację musisz pobrać przez odpowiednie narzędzie. Nie wolno Ci zgadywać.
4. ARGUMENTY JSON: Jeśli wywołujesz narzędzia, upewnij się, że składnia argumentów jest poprawnym JSON-em.

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
- Tylko fakty wynikające z wywołania narzędzi.

Pamiętaj: Jeśli narzędzie `check_availability` zwróci 'assigned', to `availability_fte` do scoringu wynosi 0.0."""

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def read_uploaded_file(uploaded_file):
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
# Streamlit UI
# ------------------------------------------------------------------------------
st.set_page_config(page_title="HR TalentMatch AI (Simple RAG)", page_icon="🤖", layout="centered")
st.title("🤖 HR TalentMatch AI (Simple Loop)")
st.caption("Wersja Standard RAG (bez LangGraph i AgentExecutor).")

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

# Sidebar
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
        if st.button(("▶ " if is_current else "☰ ") + display_name, key=f"row-{tid}", use_container_width=True):
            if not is_current:
                st.session_state.current_thread_id = tid
                st.rerun()

    with st.expander("Opcje", expanded=False):
        new_name = st.text_input("Nazwa", value=st.session_state.threads[st.session_state.current_thread_id]["name"], key="edit_name")
        if new_name != st.session_state.threads[st.session_state.current_thread_id]["name"]:
            st.session_state.threads[st.session_state.current_thread_id]["name"] = new_name
            st.rerun()
        if st.button("Usuń", key="del_thread"):
            if len(st.session_state.threads) > 1:
                del st.session_state.threads[st.session_state.current_thread_id]
                st.session_state.current_thread_id = next(iter(st.session_state.threads.keys()))
                st.rerun()

# Main Window Logic
current_thread = st.session_state.current_thread_id
thread_data = st.session_state.threads[current_thread]
messages = thread_data["messages"]

# Display Chat History
for msg in messages:
    if isinstance(msg, SystemMessage): continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"

    # Hide ToolMessages behind expanders
    if isinstance(msg, ToolMessage):
         with st.expander(f"🛠️ Wynik narzędzia: {msg.name}"):
             st.code(msg.content, language="json")
         continue

    content_to_show = msg.content
    if role == "user" and "TREŚĆ PLIKU:" in msg.content:
        content_to_show = "📄 *[Użytkownik przesłał plik do analizy]*"

    if content_to_show:
        with st.chat_message(role):
            st.markdown(content_to_show)

# File Upload Section
with st.expander("📎 Wgraj plik (CV lub RFP)", expanded=False):
    col_file, col_btn = st.columns([4, 1])
    with col_file:
        uploaded_file = st.file_uploader("Wybierz plik", type=["pdf", "docx", "txt"], label_visibility="collapsed", key=f"up_{st.session_state.uploader_key}")
    with col_btn:
        st.write("")
        if st.button("Prześlij", use_container_width=True) and uploaded_file:
            with st.spinner("Analiza..."):
                txt = read_uploaded_file(uploaded_file)
                if not txt.startswith("Błąd"):
                    sys_snippet = f"TREŚĆ PLIKU:\n{txt[:40000]}\n\nPostępuj zgodnie z instrukcjami System Prompt."
                    messages.append(HumanMessage(content=sys_snippet))

                    # Auto rename
                    visible_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
                    if len(visible_msgs) == 1:
                        thread_data["name"] = f"Plik: {uploaded_file.name[:20]}"
                        thread_data["auto_named"] = True

                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    st.error(txt)

# Chat Input & Main Loop
prompt = st.chat_input("Wpisz polecenie...")
if prompt:
    # 1. Add User Message
    messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Auto rename if first message
    visible_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(visible_msgs) == 1 and not thread_data.get("auto_named", False):
        st.session_state.threads[current_thread]["name"] = prompt[:30] + "..."
        thread_data["auto_named"] = True

    # 2. Agent Loop
    with st.chat_message("assistant"):
        placeholder = st.empty()

        # We manually loop: LLM -> ToolCalls? -> RunTools -> LLM ...
        loop_active = True
        steps = 0
        MAX_STEPS = 12

        while loop_active and steps < MAX_STEPS:
            with st.spinner(f"Analiza danych (krok {steps+1})..."):
                # Call LLM
                response_msg = llm_with_tools.invoke(messages)
                messages.append(response_msg)

                if response_msg.tool_calls:
                    # Execute tools
                    for tc in response_msg.tool_calls:
                        t_name = tc["name"]
                        t_args = tc["args"]
                        t_id = tc["id"]

                        func = tool_map.get(t_name)
                        result_content = "Tool not found"
                        if func:
                            try:
                                # Safe invocation for both Functions and StructuredTools
                                if hasattr(func, "invoke"):
                                    # Invoke method for LangChain tools
                                    result_content = str(func.invoke(t_args))
                                else:
                                    # Direct call for python functions
                                    result_content = str(func(**t_args))
                            except Exception as e:
                                result_content = f"Error executing tool: {e}"

                        messages.append(ToolMessage(content=str(result_content), tool_call_id=t_id, name=t_name))
                    steps += 1
                else:
                    # Final response (text)
                    placeholder.markdown(response_msg.content)
                    loop_active = False

        if steps >= MAX_STEPS:
            st.warning("Przekroczono limit kroków agenta.")

        st.session_state.threads[current_thread]["messages"] = messages
        st.rerun()
