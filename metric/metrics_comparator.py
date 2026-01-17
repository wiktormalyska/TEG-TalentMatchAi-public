# metric/metrics_comparator.py
import time
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, TypedDict, Annotated

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.constants import START

# Add root to sys.path to import tools
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.employees import get_employees, create_employee, delete_employee, update_employee
from tools.skills import get_skills, create_skills, delete_skill, update_skill
from tools.projects import get_project_assignments, add_project_assignment, delete_project_assignment, check_availability, delete_project, create_project, update_project, list_projects

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.ERROR) # Keep logs quiet for the benchmark

deploy_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5-nano"
api_ver = os.getenv("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview"

# FIX: Removed temperature=0 because some Azure models (like o1/preview) force temperature=1
llm = AzureChatOpenAI(
    azure_deployment=deploy_name,
    api_version=api_ver,
)

# ------------------------------------------------------------------------------
# Shared Resources (Tools & Prompts)
# ------------------------------------------------------------------------------

# We must redefine match_rfp_scoring locally to avoid importing chat_bot.py (which triggers Streamlit)
def match_rfp_scoring(
        candidate_skills: List[str],
        must_have_skills: List[str],
        nice_to_have_skills: List[str],
        availability_fte: float
) -> str:
    """
    Calculates the matching score for a candidate based on skills (Must/Nice) and availability.
    Returns a JSON string with score details and verdict.
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

tools = [
    get_employees, create_employee, delete_employee, update_employee,
    get_skills, create_skills, delete_skill, update_skill,
    get_project_assignments, add_project_assignment, delete_project_assignment, check_availability,
    match_rfp_scoring, create_project, update_project, list_projects, delete_project
]

system_prompt = "JESTEŚ NARZĘDZIEM DO TESTÓW WYDAJNOŚCIOWYCH. UŻYWAJ NARZĘDZI PRECYZYJNIE. ODPOWIADAJ KRÓTKO."

# ------------------------------------------------------------------------------
# ENGINE 1: Simple RAG (Manual Loop)
# ------------------------------------------------------------------------------
def run_simple_rag(query: str):
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
    llm_with_tools = llm.bind_tools(tools)

    # Tool mapping
    tool_map = {t.name if hasattr(t, "name") else t.__name__: t for t in tools}

    start_time = time.time()
    tool_usage_count = 0
    steps = 0
    MAX_STEPS = 10
    loop_active = True

    while loop_active and steps < MAX_STEPS:
        response_msg = llm_with_tools.invoke(messages)
        messages.append(response_msg)

        if response_msg.tool_calls:
            for tc in response_msg.tool_calls:
                t_name = tc["name"]
                t_args = tc["args"]
                t_id = tc["id"]
                tool_usage_count += 1

                func = tool_map.get(t_name)
                result_content = "Tool not found"
                if func:
                    try:
                        if hasattr(func, "invoke"):
                            result_content = str(func.invoke(t_args))
                        else:
                            result_content = str(func(**t_args))
                    except Exception as e:
                        result_content = f"Error: {e}"

                messages.append(ToolMessage(content=str(result_content), tool_call_id=t_id, name=t_name))
            steps += 1
        else:
            loop_active = False

    end_time = time.time()
    return {
        "engine": "Simple RAG",
        "time": end_time - start_time,
        "tools_used": tool_usage_count,
        "steps": steps
    }

# ------------------------------------------------------------------------------
# ENGINE 2: LangGraph (Graph RAG)
# ------------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

def run_langgraph_rag(query: str):
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    # Compile (No Checkpointer needed for single-shot benchmark)
    graph = graph_builder.compile()

    start_time = time.time()

    initial_state = {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)]}
    final_state = graph.invoke(initial_state)

    end_time = time.time()

    # Count tools by checking ToolMessages in history
    tool_usage_count = sum(1 for m in final_state['messages'] if isinstance(m, ToolMessage))

    return {
        "engine": "LangGraph",
        "time": end_time - start_time,
        "tools_used": tool_usage_count,
        "steps": len(final_state['messages']) # Rough proxy for steps
    }

# ------------------------------------------------------------------------------
# Main Execution & Visualization
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "Wylistuj wszystkich pracowników w systemie.",
        "Znajdź pracowników z umiejętnością 'Python'.",
        "Kto jest przypisany do projektu 'Internal'?",
        "Jakie umiejętności ma Jan Kowalski?",
        "Sprawdź dostępność pracownika Jan Kowalski (czy ma bench).",
        "Wypisz wszystkie projekty dostępne w bazie.",
        "Znajdź kandydatów znających 'Java' oraz 'AWS'.",
        "Czy Anna Nowak jest przypisana do jakiegokolwiek projektu?",
        "Jakie skille są zdefiniowane w systemie? Wylistuj je.",
        "Znajdź pracowników z umiejętnością 'React' i sprawdź ich dostępność."
    ]

    results = []

    print("🚀 Rozpoczynanie benchmarku...")

    for i, q in enumerate(test_queries):
        print(f"\n[{i+1}/{len(test_queries)}] Testowanie: '{q}'")

        # Test Simple RAG
        print("   Running Simple RAG...", end="", flush=True)
        try:
            res_simple = run_simple_rag(q)
            res_simple["query_id"] = i + 1
            results.append(res_simple)
            print(f" Done ({res_simple['time']:.2f}s)")
        except Exception as e:
            print(f" Error: {e}")

        # Test LangGraph
        print("   Running LangGraph...", end="", flush=True)
        try:
            res_graph = run_langgraph_rag(q)
            res_graph["query_id"] = i + 1
            results.append(res_graph)
            print(f" Done ({res_graph['time']:.2f}s)")
        except Exception as e:
             print(f" Error: {e}")

    # Data Processing
    df = pd.DataFrame(results)

    # Plotting
    if not df.empty:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Metric 1: Execution Time
            pivot_time = df.pivot(index="query_id", columns="engine", values="time")
            pivot_time.plot(kind="bar", ax=axes[0], color=["#FF6F61", "#6B5B95"])
            axes[0].set_title("Czas Wykonania (s)")
            axes[0].set_ylabel("Sekundy")
            axes[0].set_xlabel("ID Zapytania")
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)

            # Metric 2: Tool Usage (Calls)
            pivot_tools = df.pivot(index="query_id", columns="engine", values="tools_used")
            pivot_tools.plot(kind="bar", ax=axes[1], color=["#88B04B", "#F7CAC9"])
            axes[1].set_title("Liczba Wywołań Narzędzi")
            axes[1].set_ylabel("Liczba Tool Calli")
            axes[1].set_xlabel("ID Zapytania")
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            output_path = ROOT / "search" / "comparison_results.png"

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_path)
            print(f"\n📊 Wykres zapisano w: {output_path}")
        except Exception as e:
            print(f"\n⚠️ Błąd generowania wykresu: {e}")
    else:
        print("\n⚠️ Brak wyników do wygenerowania wykresu.")

    print("\n✅ Benchmark zakończony.")
    print(df)
