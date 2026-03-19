"""
Dynamic Planner — Creates new retrieval tasks from critic's coverage gaps.

Replaces ad-hoc refinement queries with proper task creation,
so the retry loop goes through the full planning pipeline.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DYNAMIC_PLANNER_PROMPT = """You are a dynamic planning agent. A critic found coverage gaps in the current answer. Your job is to create NEW retrieval tasks to fill those gaps.

## Input
You receive:
- The original question
- Coverage gaps identified by the critic
- Previously executed tasks (to avoid duplication)

## Your Job
Create 1-3 NEW retrieval tasks that specifically target the missing information.

## Constraints
- Each task must be action-oriented: "Retrieve...", "Find evidence for..."
- Do NOT repeat tasks that were already executed
- Each task must directly address a specific coverage gap
- Be precise — target the exact missing mechanism or concept

## Output JSON Schema
{
  "new_tasks": [
    {"task_id": "<next_id>", "description": "<action-oriented retrieval task>"}
  ],
  "rationale": "<brief explanation of why these tasks fill the gaps>"
}

Output valid JSON only.
"""


def revise_plan(query: str, coverage_gaps: list, existing_tasks: list = None) -> dict:
    """
    Generate new retrieval tasks based on coverage gaps.

    Args:
        query: The original user question
        coverage_gaps: List of missing aspects from the critic
        existing_tasks: Previously executed tasks (to avoid duplication)

    Returns:
        dict with new_tasks list
    """
    # Determine next task_id
    if existing_tasks:
        max_id = max(t.get("task_id", 0) for t in existing_tasks if isinstance(t.get("task_id"), int))
    else:
        max_id = 0

    user_message = json.dumps({
        "question": query,
        "coverage_gaps": coverage_gaps,
        "existing_tasks": existing_tasks or [],
        "next_task_id_start": max_id + 1,
    }, indent=2)

    try:
        from utils.llm_client import get_json_response
        result = get_json_response(
            messages=[
                {"role": "system", "content": DYNAMIC_PLANNER_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=512
        )

        new_tasks = result.get("new_tasks", [])
        if not new_tasks:
            return {"tasks": []}

        return {"tasks": new_tasks}

    except json.JSONDecodeError as e:
        print(f"Error parsing dynamic plan JSON: {e}")
        # Fallback: create tasks directly from gaps
        fallback_tasks = []
        for i, gap in enumerate(coverage_gaps, max_id + 1):
            fallback_tasks.append({
                "task_id": i,
                "description": f"Retrieve evidence about: {gap}"
            })
        return {"tasks": fallback_tasks}
    except Exception as e:
        print(f"Error in dynamic planning: {e}")
        return {"tasks": []}
