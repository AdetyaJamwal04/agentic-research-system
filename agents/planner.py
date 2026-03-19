import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)

PLANNER_SYSTEM_PROMPT = """You are a retrieval planning agent. You receive a structured query analysis and produce a JSON plan of specific retrieval tasks.

## Input
You will receive a JSON analysis containing:
- `original_query`: the user's question
- `refined_query`: a clearer restatement
- `intent`, `entities`, `key_concepts`, `domains`, `scope`

Use ALL of these fields to inform your task planning.

## Strategy
1. Identify the exact phenomenon the user is asking about.
2. Use `key_concepts` and `entities` to determine the specific mechanisms or causes.
3. Create 2-3 action-oriented retrieval tasks — each must describe WHAT to retrieve, not WHAT the concept is.

## Constraints
- Each task must be an action: "Retrieve explanations of..." / "Find evidence for..."
- Do NOT generate conceptual statements like "How X works" — generate retrieval actions.
- Every task must pass this test: "Would searching for this directly retrieve content explaining the user's phenomenon?"
- Avoid generic background retrieval. Target the specific mechanisms.

## Example
Input query: "Why does gradient descent zigzag?"
BAD tasks:
- "How learning rate affects gradient descent" (conceptual, too broad)
- "Local minima vs saddle points" (tangential)

GOOD tasks:
- "Retrieve explanations of how elongated loss surfaces with different curvatures cause gradient descent to oscillate perpendicular to the optimal path"
- "Find evidence for how learning rate magnitude relative to surface curvature amplifies zigzag oscillation"

## Output JSON Schema
{
  "tasks": [
    {"task_id": 1, "description": "<action-oriented retrieval task>"},
    {"task_id": 2, "description": "<action-oriented retrieval task>"}
  ]
}

Output valid JSON only. No markdown, no extra text.
"""

def create_plan(analysis):
    """
    Takes the query analysis dict and generates a structured JSON retrieval plan.
    """
    try:
        if isinstance(analysis, dict):
            user_input = json.dumps(analysis, indent=2)
        else:
            user_input = str(analysis)

        from utils.llm_client import get_json_response
        plan = get_json_response(
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=512
        )
        return plan
    except json.JSONDecodeError as e:
        print(f"Error parsing plan JSON: {e}")
        query = analysis.get("original_query", str(analysis)) if isinstance(analysis, dict) else str(analysis)
        return {"tasks": [{"task_id": 1, "description": f"Retrieve information about: {query}"}]}
    except Exception as e:
        print(f"Error creating plan: {e}")
        query = analysis.get("original_query", str(analysis)) if isinstance(analysis, dict) else str(analysis)
        return {"tasks": [{"task_id": 1, "description": f"Retrieve information about: {query}"}]}
