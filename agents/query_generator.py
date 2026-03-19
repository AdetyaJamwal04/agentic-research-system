import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

QUERY_GENERATOR_SYSTEM_PROMPT = """You are a Search Query Generator. Convert retrieval tasks into precise, phenomenon-specific web search queries.

## Input
You will receive a JSON plan with tasks, each having a `task_id` and `description`.

## Query Crafting Rules
- Preserve the specificity of each task — do NOT generalize into broad topic queries.
- Include the specific phenomenon or behavior from the task description.
- Use technical terms and domain-specific language.
- Each query should retrieve content that DIRECTLY explains the mechanism, not general overviews.
- Generate 1-2 queries per task. Quality over quantity.
- Each query: 4-10 words, directly usable in a search engine.

## Example
Task: "Retrieve explanations of how elongated loss surfaces cause gradient descent to oscillate"
BAD queries: "gradient descent", "loss surface optimization"
GOOD queries: "gradient descent zigzag elongated contours curvature", "oscillation gradient descent ill-conditioned Hessian"

## Output JSON Schema
{
  "task_queries": [
    {"task_id": 1, "queries": ["<specific_query_1>", "<specific_query_2>"]},
    {"task_id": 2, "queries": ["<specific_query_1>"]}
  ]
}

Output valid JSON only. No markdown, no extra text.
"""

REFINEMENT_SYSTEM_PROMPT = """You are a Search Query Generator in refinement mode. The previous answer was insufficient. You receive the original question and a list of missing aspects identified by a critic.

Generate 2-3 targeted search queries specifically to fill the identified gaps. Do NOT repeat earlier queries — focus only on the missing information.

## Output JSON Schema
{
  "task_queries": [
    {"task_id": "refinement", "queries": ["<targeted_query_1>", "<targeted_query_2>"]}
  ]
}

Output valid JSON only. No markdown, no extra text.
"""


def generate_queries(task_plan, api_key: str = None) -> dict:
    """
    Takes the planner's JSON task plan and generates task-mapped search queries.
    """
    try:
        api_key = api_key or os.getenv("GROQ_API_KEY")

        if isinstance(task_plan, dict):
            plan_text = json.dumps(task_plan, indent=2)
        else:
            plan_text = str(task_plan)

        from utils.llm_client import get_json_response
        queries = get_json_response(
            messages=[
                {"role": "system", "content": QUERY_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": plan_text},
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return queries
    except json.JSONDecodeError as e:
        print(f"Error parsing generated queries JSON: {e}")
        return {"task_queries": [], "error": str(e)}
    except Exception as e:
        print(f"Error generating queries: {e}")
        return {"task_queries": [], "error": str(e)}


def generate_refinement_queries(original_query: str, missing_aspects: list, api_key: str = None) -> dict:
    """
    Generates targeted queries to fill gaps identified by the critic.
    """
    try:
        api_key = api_key or os.getenv("GROQ_API_KEY")

        user_input = json.dumps({
            "original_query": original_query,
            "missing_aspects": missing_aspects
        }, indent=2)

        from utils.llm_client import get_json_response
        return get_json_response(
            messages=[
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.3,
            max_tokens=512
        )
    except Exception as e:
        print(f"Error generating refinement queries: {e}")
        return {"task_queries": [{"task_id": "refinement", "queries": missing_aspects}]}
