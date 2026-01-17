from pathlib import Path
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools.employees import get_employees

# Konfiguracja log Pipes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
            logging.FileHandler("system.log"),
        logging.StreamHandler()
    ]
)

system_logger = logging.getLogger("system")
ai_logger = logging.getLogger("ai")

ai_handler = logging.FileHandler('ai.log')
ai_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
ai_logger.addHandler(ai_handler)
ai_logger.setLevel(logging.INFO)

current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
project_root = current_dir.parent
env_path = project_root / '.env'

system_logger.info(f"Loading .env from: {env_path.resolve()}")
load_dotenv(env_path)

deploy_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_ver = os.getenv("AZURE_OPENAI_API_VERSION")

if not deploy_name or not api_ver:
    # Fallback na wypadek gdyby ktoś zapomniał dodać do .env
    system_logger.warning("Warning: Missing Azure config in .env, using defaults.")
    deploy_name = "gpt-5-nano"
    api_ver = "2025-01-01-preview"

system_logger.info(f"Using Azure Deployment: {deploy_name}, API Version: {api_ver}")

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

llm = AzureChatOpenAI(
    azure_deployment=deploy_name,
    api_version=api_ver,
)

from tools.employees import get_employees, create_employee, delete_employee, update_employee
from tools.skills import get_skills, create_skills, delete_skill, update_skill

tools = [get_employees, create_employee, delete_employee, update_employee, get_skills, create_skills, delete_skill, update_skill] #  Tu dodaj toole jakie są

llm_with_tools = llm.bind_tools(tools)

from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

system_prompt = """"Jesteś inteligentnym asystentem HR do wyszukiwania i zarządzania pracownikami firmy. Masz dostęp do:

- Pełnej listy pracowników z ich umiejętnościami (get_employees)
- Bazy umiejętności (get_skills) 
- Możliwości dodawania nowych umiejętności (create_skills)
- Usuwania umiejętności (delete_skill)

Główne zadania:
1. Na podstawie zapytań użytkownika wyszukuj odpowiednich pracowników według ich umiejętności, doświadczenia, roli itp.
2. Polecaj najlepsze dopasowania do konkretnych potrzeb (np. "Kto zna Python i SQL?", "Znajdź developera React")
3. Zarządzaj umiejętnościami pracowników gdy użytkownik o to poprosi
4. Odpowiadaj zwięźle i konkretnie, zawsze podając imiona/nazwiska pasujących osób
5. Jeśli nie znajdziesz idealnego dopasowania, zasugeruj najbliższe opcje

Zawsze używaj narzędzi do pobierania aktualnych danych z bazy."""
config = {"configurable": {"thread_id": "1"}}

system_logger.info("Chat started (LangGraph). Type 'exit' to quit.")

initial_state = {"messages": [SystemMessage(content=system_prompt)]}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    events = graph.stream(
        {"messages": [("user", user_input)]},
        config,
        stream_mode="values"
    )

    for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]
            # AI RESPONSE DLA UI + CLI
            if last_msg.type == "ai" and last_msg.content:
                ai_logger.info(last_msg.content)
                print(f"AI: {last_msg.content}")
            # TOOL CALLS DLA SYSTEM LOG
            elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                system_logger.info(f"Tool calls: {[tc['name'] for tc in last_msg.tool_calls]}")
