import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# __define-ocg__ — query analysis agent setup
# __define-pcb__ — llm-based query understanding logic

QUERY_ANALYSER_SYSTEM_PROMPT = """You are a Query Analysis Agent. Analyse the user's query and produce a structured JSON analysis for a downstream Planning Agent.

## Responsibilities
1. **Intent Classification** — Classify as: `factual`, `analytical`, `procedural`, `exploratory`, or `conversational`.
2. **Entity Extraction** — Extract key entities and technical terms explicitly mentioned in the query.
3. **Key Concepts** — Infer deeper concepts, mechanisms, and technical ideas related to the query that are NOT explicitly mentioned but essential to a complete answer (e.g., for "Why does gradient descent zigzag?" → loss surface, curvature, oscillation, condition number).
4. **Ambiguity Detection** — Flag ambiguous terms or assumptions. Return empty list if none.
5. **Scope Assessment** — Determine depth: `brief`, `moderate`, or `detailed`.
6. **Domain Identification** — Identify knowledge domains (e.g., programming, science, history).
7. **Contextual Requirements** — What sources/tools are needed (e.g., web search, document retrieval).

## Output JSON Schema
{
  "original_query": "<verbatim query>",
  "intent": "<factual|analytical|procedural|exploratory|conversational>",
  "entities": ["<explicit_entity1>", "<explicit_entity2>"],
  "key_concepts": ["<inferred_concept1>", "<inferred_concept2>", "<inferred_concept3>"],
  "ambiguities": ["<ambiguity1>"],
  "scope": "<brief|moderate|detailed>",
  "domains": ["<domain1>"],
  "contextual_requirements": ["<requirement1>"],
  "refined_query": "<clearer restatement preserving original intent>"
}

## Rules
- Output valid JSON only. No markdown, no extra text.
- `entities` = what the user explicitly mentions. `key_concepts` = deeper ideas needed for a complete answer.
- Be concise — the Planning Agent needs actionable structure, not explanations.
"""
def analyse_query(user_query: str, api_key: str = os.getenv("GROQ_API_KEY")) -> dict:
    """
    Sends the user query to llama3.3-70B-Versatile via Groq
    and returns a structured analysis as a dictionary.
    """
    from utils.llm_client import get_json_response
    try:
        analysis = get_json_response(
            messages=[
                {"role": "system", "content": QUERY_ANALYSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return analysis
        print(f"Error parsing query analysis JSON: {e}")
        return {"original_query": user_query, "error": "Failed to parse analysis"}
    except Exception as e:
        print(f"Error analysing query: {e}")
        return {"original_query": user_query, "error": str(e)}
    



