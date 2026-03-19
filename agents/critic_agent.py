import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key= os.getenv("GROQ_API_KEY")
)

CRITIC_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated answers. Evaluate whether the answer correctly, completely, and deeply explains the user's question.

## Scoring Criteria
1. **Correctness** — Are the claims factually accurate? No hallucinations or contradictions.
2. **Completeness** — Does the answer explain ALL key mechanisms/causes? Penalize if core explanations are missing.
3. **Depth** — Does the answer explain WHY and HOW, not just WHAT? Surface-level answers score lower.
4. **Evidence Usage** — Does the answer reference and integrate the evidence it was given?

## Scoring Scale
- 1-3: Major errors, missing most key information, or incoherent.
- 4-5: Partially correct but missing important mechanisms or explanations.
- 6-7: Mostly correct and complete, but lacks depth on some aspects.
- 8-10: Accurate, comprehensive, and well-reasoned with strong evidence usage.

## Verdict Rule
- Score >= 8 → verdict: "sufficient"
- Score < 8 → verdict: "insufficient"

## Missing Aspects
When verdict is "insufficient", identify specific concepts, mechanisms, or explanations that are missing from the answer. Be concrete — name the exact topics that need more research.

## Output JSON Schema
{
  "score": <1-10>,
  "verdict": "<sufficient|insufficient>",
  "reason": "<one-sentence explanation>",
  "missing_aspects": ["<specific missing concept or mechanism>"]
}

Output valid JSON only. Return empty list for missing_aspects if verdict is sufficient.
"""

def critique_answer(query, answer):

    user_message = f"""User Question:
{query}

Answer:
{answer}"""

    try:
        from utils.llm_client import get_json_response
        return get_json_response(
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=256
        )
    except json.JSONDecodeError as e:
        print(f"Error parsing critic JSON: {e}")
        return {
            "score": 0,
            "verdict": "insufficient",
            "reason": "Could not parse critic output"
        }
    except Exception as e:
        print(f"Error in critic agent: {e}")
        return {
            "score": 0,
            "verdict": "insufficient",
            "reason": f"Critic error: {e}"
        }