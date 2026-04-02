"""
LangGraph implementation of Planner-Executor Agent
Converts LangChain logic to LangGraph workflow with direct tool calling
"""

import json
import re
import os
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    # Load project-level .env so GROQ_API_KEY is available without manual export.
    load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """State for Planner-Executor Agent"""
    goal: str
    plan: List[dict]
    current_step: int
    results: List[dict]
    full_context: str


# ═══════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = {
    "calculate_tables_chairs": {
        "description": "Calculate number of tables and chairs needed for an event",
        "func": lambda people, chairs_per_table=6: {
            "tables": max(1, people // chairs_per_table),
            "chairs": people,
            "setup_notes": f"For {people} people: {people // chairs_per_table} tables with {chairs_per_table} chairs each"
        }
    },
    "get_weather": {
        "description": "Get weather information for a location",
        "func": lambda city="outdoor": {
            "city": city,
            "temperature": "72°F",
            "condition": "Partly Cloudy",
            "humidity": "65%",
            "wind": "10 km/h",
            "recommendation": "Perfect conditions for an outdoor event"
        }
    },
    "calculate_ticket_price": {
        "description": "Calculate average ticket price based on budget and attendees",
        "func": lambda people, budget=0: {
            "attendees": people,
            "total_budget": budget or (people * 25),
            "avg_price": (budget or (people * 25)) / people if people > 0 else 0,
            "pricing_note": f"Average ticket price: ${(budget or (people * 25)) / people:.2f}"
        }
    },
    "search_venues": {
        "description": "Search for available venues",
        "func": lambda capacity=150: {
            "venues_found": 3,
            "venues": [
                {"name": "Grand Pavilion", "capacity": 200, "price_per_person": 15},
                {"name": "Garden Terrace", "capacity": 250, "price_per_person": 12},
                {"name": "Outdoor Park Center", "capacity": 300, "price_per_person": 10}
            ],
            "recommendation": f"All venues support {capacity} people comfortably"
        }
    }
}

TOOL_ARG_MAP = {
    "calculate_tables_chairs": ["people"],
    "get_weather": ["city"],
    "calculate_ticket_price": ["people", "budget"],
    "search_venues": ["capacity"],
}


# ═══════════════════════════════════════════════════════════════════════════
# LLM SETUP
# ═══════════════════════════════════════════════════════════════════════════

def get_llm():
    """Initialize LLM with GROQ API"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n⚠️  GROQ_API_KEY not set. Using demo mode (simulated responses).\n")
        return None
    # Allow overriding via .env; default to a currently supported Groq model.
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)


PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools and their arguments:
  - calculate_tables_chairs(people: int) → Calculate tables and chairs needed
  - get_weather(city: str) → Get weather for a location  
  - calculate_ticket_price(people: int, budget: int) → Calculate average ticket price
  - search_venues(capacity: int) → Find suitable venues

Use null for tool/args on synthesis or writing steps.
Return ONLY a valid JSON array. No markdown, no explanation."""


# ═══════════════════════════════════════════════════════════════════════════
# DEMO MODE (For when GROQ_API_KEY is not set)
# ═══════════════════════════════════════════════════════════════════════════

def get_demo_plan(goal: str) -> list:
    """Return a demo plan when LLM is not available"""
    return [
        {"step": 1, "description": "Analyze event requirements and constraints", "tool": None, "args": None},
        {"step": 2, "description": "Calculate tables and chairs needed", "tool": "calculate_tables_chairs", "args": {"people": 150}},
        {"step": 3, "description": "Check weather forecast for outdoor event", "tool": "get_weather", "args": {"city": "outdoor"}},
        {"step": 4, "description": "Calculate average ticket price", "tool": "calculate_ticket_price", "args": {"people": 150}},
        {"step": 5, "description": "Search for suitable venues", "tool": "search_venues", "args": {"capacity": 150}},
        {"step": 6, "description": "Summarize event planning recommendations", "tool": None, "args": None}
    ]


def get_demo_response(description: str, context: str) -> str:
    """Return demo responses for synthesis steps"""
    if "analyze" in description.lower() or "requirement" in description.lower():
        return "✅ Requirements analyzed: Event for 150 people, outdoor setting with multiple planning tasks including venue setup, weather planning, and budgeting."
    elif "summarize" in description.lower():
        return "📋 Event Summary: Based on the analysis, we recommend booking a venue with 200+ capacity, arranging 25 tables with 6 chairs each, checking weather 5 days before, and setting ticket price at $25-30 per person."
    else:
        return f"✅ Step completed: {description}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool and return result"""
    if tool_name not in TOOLS:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        tool = TOOLS[tool_name]
        result = tool["func"](**args) if args else tool["func"]()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def safe_args(tool_name: str, raw_args: dict) -> dict:
    """Ensure correct argument names for tools"""
    if not raw_args:
        return {}
    
    expected_args = TOOL_ARG_MAP.get(tool_name, [])
    if not expected_args:
        return raw_args
    
    # Check if all expected args are present
    if all(arg in raw_args for arg in expected_args):
        return raw_args
    
    # Try to map provided args to expected args
    provided_values = list(raw_args.values())
    remapped = {}
    for i, expected_arg in enumerate(expected_args):
        if i < len(provided_values):
            remapped[expected_arg] = provided_values[i]
    
    return remapped if remapped else raw_args


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ═══════════════════════════════════════════════════════════════════════════

def planner_node(state: AgentState) -> AgentState:
    """PLANNER NODE: Generate a plan for the given goal"""
    print(f"\n{'='*60}")
    print(f"🎯 PLANNER NODE - Generating plan for goal:")
    print(f"'{state['goal']}'")
    print(f"{'='*60}\n")
    
    llm = get_llm()
    
    if llm is None:
        print("📝 [DEMO MODE] Generating sample plan...\n")
        plan = get_demo_plan(state['goal'])
    else:
        # Call LLM to generate plan
        plan_resp = llm.invoke([
            SystemMessage(content=PLAN_SYSTEM),
            HumanMessage(content=state['goal'])
        ])
        
        # Parse response
        raw_text = plan_resp.content if isinstance(plan_resp.content, str) else plan_resp.content[0].get("text", "")
        clean_json = re.sub(r"```json|```", "", raw_text).strip()
        
        try:
            plan = json.loads(clean_json)
        except json.JSONDecodeError:
            print(f"❌ Failed to parse plan. Using fallback...\n")
            plan = get_demo_plan(state['goal'])
    
    # Display plan
    print(f"✅ Generated {len(plan)} steps:\n")
    for s in plan:
        tool_info = f" → {s.get('tool')}" if s.get('tool') else ""
        print(f"   Step {s['step']}: {s['description']}{tool_info}")
    
    print()
    
    return {
        "goal": state["goal"],
        "plan": plan,
        "current_step": 0,
        "results": [],
        "full_context": state.get("full_context", "")
    }


def executor_node(state: AgentState) -> Command:
    """EXECUTOR NODE: Execute one step from the plan"""
    plan = state["plan"]
    current_step = state["current_step"]
    results = state["results"]
    
    # Check if we're done
    if current_step >= len(plan):
        print(f"\n{'='*60}")
        print(f"✅ ALL STEPS COMPLETED")
        print(f"{'='*60}\n")
        return Command(
            update={
                "goal": state["goal"],
                "plan": plan,
                "current_step": current_step,
                "results": results,
                "full_context": state.get("full_context", "")
            },
            goto=END
        )
    
    # Get current step
    step = plan[current_step]
    step_num = step["step"]
    description = step["description"]
    tool_name = step.get("tool")
    tool_args = step.get("args") or {}
    
    print(f"\n{'─'*60}")
    print(f"⚡ EXECUTOR NODE - Step {step_num}/{len(plan)}")
    print(f"{'─'*60}")
    print(f"📋 Task: {description}")
    
    # Execute step
    if tool_name and tool_name in TOOLS:
        print(f"🔧 Using tool: {tool_name}")
        print(f"   Args: {tool_args}")
        corrected_args = safe_args(tool_name, tool_args)
        result = execute_tool(tool_name, corrected_args)
        print(f"   ✓ Result: {result[:100]}...")
    else:
        # Synthesis step - use LLM
        print(f"🧠 Synthesis step (no tool)")
        llm = get_llm()
        context = "\n".join([f"Step {r['step']}: {r['result'][:80]}" for r in results])
        
        if llm is None:
            # Demo mode
            result = get_demo_response(description, context)
        else:
            # Real LLM call
            prompt = f"""Based on the goal and previous results, complete this step:

Step: {description}

Previous Results:
{context if context else "None yet"}

Provide a clear, concise response."""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content
        
        print(f"   ✓ Result: {result[:100]}...")
    
    # Store result
    results.append({
        "step": step_num,
        "description": description,
        "result": str(result),
        "tool": tool_name
    })
    
    # Increment step
    next_step = current_step + 1
    
    # Continue to next step
    return Command(
        update={
            "goal": state["goal"],
            "plan": plan,
            "current_step": next_step,
            "results": results,
            "full_context": state.get("full_context", "")
        },
        goto="executor"
    )


# ═══════════════════════════════════════════════════════════════════════════
# BUILD & RUN GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def build_graph():
    """Build and return the LangGraph workflow"""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    
    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    
    # Compile graph
    compiled_graph = graph.compile()
    
    return compiled_graph


def run_agent(goal: str):
    """Run the Planner-Executor agent"""
    print(f"\n{'#'*60}")
    print(f"# PLANNER-EXECUTOR AGENT (LangGraph)")
    print(f"{'#'*60}")
    print(f"Goal: {goal}\n")
    
    # Build graph
    graph = build_graph()
    
    # Initial state
    initial_state = {
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "results": [],
        "full_context": ""
    }
    
    # Run graph
    final_state = graph.invoke(initial_state)
    
    # Display final results
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS")
    print(f"{'='*60}\n")
    
    for result in final_state["results"]:
        print(f"Step {result['step']}: {result['description']}")
        if result.get('tool'):
            print(f"  Tool: {result['tool']}")
        print(f"  Result: {result['result'][:200]}...")
        print()
    
    return final_state


if __name__ == "__main__":
    # Test goal
    goal = "Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize."
    result = run_agent(goal)
