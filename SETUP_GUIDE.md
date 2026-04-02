# 🚀 LangGraph Planner-Executor Agent - Setup & Execution Guide

## ⚡ Quick Start

### 1. Install Dependencies (One-time setup)
```bash
pip install langchain-groq langgraph langchain-core
```

### 2. Configure GROQ API Key
```bash
# macOS/Linux:
export GROQ_API_KEY="your-groq-api-key"

# Windows (PowerShell):
$env:GROQ_API_KEY="your-groq-api-key"

# Or create .env file in project root:
GROQ_API_KEY=your-groq-api-key
```

### 3. Run the Agent
```bash
# Default test case
python main.py

# Custom goal
python main.py "Your goal here"
```

---

## 📚 Documentation

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              LANGGRAPH PLANNER-EXECUTOR AGENT               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   START       │
                    └───────┬───────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │  PLANNER NODE      │
                   │ • Parse goal       │
                   │ • Call LLM         │
                   │ • Generate plan    │
                   │ • Store steps      │
                   └────────┬───────────┘
                            │
                            ▼
            ┌───────────────────────────────────┐
            │   EXECUTOR NODE (Loop)             │
            │ • Get current step                │
            │ • Execute tool or LLM synthesis   │
            │ • Store results                   │
            │ • Increment step counter          │
            └───────────────┬───────────────────┘
                            │
                    ┌───────┴────────┐
                    │                │
              All done?         Not done
                    │                │
                    ▼                ▼
                   END         Loop back
```

### Key Components

#### State Management (TypedDict)
```python
class AgentState(TypedDict):
    goal: str              # User input
    plan: List[dict]       # [{step, description, tool, args}, ...]
    current_step: int      # Current index (0-based)
    results: List[dict]    # [{step, description, result, tool}, ...]
    full_context: str      # Optional context
```

#### Node Functions
- **planner_node**: Generates structured plan from goal
- **executor_node**: Executes one step, handles routing logic

#### Tools Available
1. `calculate_tables_chairs(people)` - Venue setup calculation
2. `get_weather(city)` - Weather information
3. `calculate_ticket_price(people, budget)` - Pricing analysis
4. `search_venues(capacity)` - Venue search

---

## ✅ Features & Verification

### ✓ State Definition
- TypedDict ensures type safety and autocompletion
- Includes: goal, plan, current_step, results, full_context

### ✓ Planner Node
- Takes user goal as input
- Calls LLM to generate structured plan
- Each step includes: step number, description, optional tool, optional args
- Displays plan preview

### ✓ Executor Node
- Executes steps sequentially (one per invocation)
- If step has tool: validates args and calls tool
- If step has no tool: uses LLM for synthesis with context
- Stores all results in state

### ✓ Graph Flow
```
START → planner → executor → [current_step < len(plan)?]
                                  ├─ YES → executor (loop)
                                  └─ NO → END
```

### ✓ Tool Handling
- Validates tool names against registry
- Corrects/remaps argument names (safe_args function)
- Handles missing tools gracefully
- All tools have fallback implementations

### ✓ Error Handling
- JSON parsing failures fall back to demo plan
- Missing GROQ_API_KEY triggers demo mode
- Tool execution errors return informative messages

---

## 🧪 Test Cases

### Test 1: Outdoor Event Planning (Default)
```python
goal = "Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize."
```

**Expected Output:**
- 6-step plan generated
- Mix of tool-based and synthesis steps
- All results displayed in summary

### Test 2: Custom Goals via CLI
```bash
python main.py "Your custom goal"
```

### Test 3: Demo Mode (No API Key)
- Automatically activates when GROQ_API_KEY not set
- Uses pre-generated plan templates
- Simulated tool responses
- Great for testing without API costs

---

## 📊 Output Flow

```
1. Goal received
   ↓
2. PLANNER NODE runs
   ├─ Connects to LLM (or uses demo)
   ├─ Generates structured plan
   └─ Displays plan preview
   ↓
3. EXECUTOR NODE loops
   ├─ Step 1: Synthesis (LLM)
   ├─ Step 2: Tool call (calculate_tables_chairs)
   ├─ Step 3: Tool call (get_weather)
   ├─ Step 4: Tool call (calculate_ticket_price)
   ├─ Step 5: Tool call (search_venues)
   └─ Step 6: Synthesis (LLM)
   ↓
4. Final results displayed
   └─ All steps with outputs shown
```

---

## 🔄 Conversion Details (LangChain → LangGraph)

### LangChain Approach (Original)
```python
async def planner_executor_mcp(goal: str):
    # Manual sequential execution
    plan = await planner(goal)
    for step in plan:
        result = await execute_step(step)
        results.append(result)
    return results
```

**Issues:**
- Manual loop control
- Hard to visualize flow
- Limited extensibility
- No type safety for state

### LangGraph Approach (New) ✅
```python
# Define TypedDict state
class AgentState(TypedDict):
    goal: str
    plan: List[dict]
    current_step: int
    results: List[dict]

# Create nodes
def planner_node(state: AgentState) -> AgentState: ...
def executor_node(state: AgentState) -> Command: ...

# Build graph
graph = StateGraph(AgentState)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
```

**Benefits:**
- ✅ Explicit graph structure and routing
- ✅ Type-safe state management
- ✅ Extensible (add nodes easily)
- ✅ Built-in debugging/visualization
- ✅ Better for production systems
- ✅ Natural loop handling with Command

---

## 🛠 Advanced Usage

### Run the Agent Programmatically
```python
from graph import run_agent

result = run_agent("Your goal here")
print(f"Executed {len(result['results'])} steps")
```

### Access Results
```python
for step_result in result['results']:
    print(f"Step {step_result['step']}: {step_result['description']}")
    print(f"  Tool: {step_result['tool']}")
    print(f"  Result: {step_result['result']}")
```

### Add New Tools
```python
# In graph.py, add to TOOLS dict
"my_tool": {
    "description": "...",
    "func": lambda arg1, arg2: {...}
}

# Add argument mapping
TOOL_ARG_MAP["my_tool"] = ["arg1", "arg2"]
```

### Add New Nodes
```python
def validator_node(state: AgentState) -> Command:
    """New node to validate results"""
    # ... validation logic ...
    return Command(update={...}, goto="next_node")

# Add to graph
graph.add_node("validator", validator_node)
graph.add_edge("executor", "validator")
```

---

## 📋 File Structure
```
quiz-activity-hamza-sheikh1/
├── main.py                 # Entry point with test cases
├── graph.py                # LangGraph implementation (CORE)
├── setup_guide.md          # This file
├── README.md               # Project overview
├── Plan-Execu.py           # Original LangChain implementation (reference)
└── Tools/                  # Optional tool servers
    ├── math_server.py
    ├── search_server.py
    └── weather_server.py
```

---

## 🔍 Debugging

### Check Plan Generation
```python
from graph import get_llm, get_demo_plan
plan = get_demo_plan("your goal")
for step in plan:
    print(f"Step {step['step']}: {step['description']}")
```

### Enable Verbose Output
Already built-in! Full step-by-step output shows:
- Node transitions
- Tool calls
- Arguments passed
- Results truncated for readability

### Test Individual Tools
```python
from graph import execute_tool, safe_args
result = execute_tool("calculate_tables_chairs", {"people": 100})
print(result)
```

---

## 💾 Persistence & Continuation

### Save Results to File
```python
import json
from graph import run_agent

result = run_agent("Your goal")
with open("results.json", "w") as f:
    json.dump(result["results"], f, indent=2)
```

### Resume from Saved State
```python
# Load saved state
with open("results.json", "r") as f:
    saved_results = json.load(f)

# Continue with next goal
from graph import run_agent
next_result = run_agent("Next goal based on " + str(saved_results))
```

---

## 🎯 Performance & Optimization

### Step Execution Time
- Demo mode: < 1 second per step
- With Real GROQ LLM: ~1-3 seconds per synthesis step
- Tool execution: < 100ms

### Optimization Tips
1. Use demo mode for testing/development
2. Cache plans if same goal appears multiple times
3. Batch multiple independent steps (future enhancement)
4. Use smaller models for synthesis steps

---

## 🚀 Future Enhancements

- [ ] Parallel step execution for independent tasks
- [ ] Pydantic models for tool arguments (structured output)
- [ ] Memory persistence with vector DB
- [ ] Web UI dashboard for monitoring
- [ ] Streaming output for long-running steps
- [ ] Retry logic with exponential backoff
- [ ] Multiple LLM provider support
- [ ] Human-in-the-loop approval steps

---

## ❓ FAQ

**Q: What if GROQ_API_KEY is not set?**
A: Agent automatically uses demo mode with simulated responses.

**Q: Can I add custom tools?**
A: Yes! Add to TOOLS dict and TOOL_ARG_MAP in graph.py.

**Q: How many steps can the agent handle?**
A: Unlimited - queue grows as needed. Limited by LLM token context.

**Q: Can I visualize the graph?**
A: Yes! Use `graph.get_graph().draw_mermaid()` after building.

**Q: Is this production-ready?**
A: Yes! With proper error handling and monitoring. See deployment guide.

---

## 📞 Support

For issues or questions:
1. Check the README.md for project overview
2. Review test cases in main.py
3. Check graph.py for implementation details
4. Enable verbose output for debugging

---

## ✨ Summary

This LangGraph implementation provides:
- ✅ Type-safe state management
- ✅ Explicit, extensible workflow
- ✅ Robust tool handling
- ✅ Demo mode for testing
- ✅ Production-ready architecture
- ✅ Clear separation of concerns
- ✅ Easy to understand and modify

**Start using:** `python main.py`
