"""
Premise Validator — Checks if the user's query contains a false or questionable premise.

Inserted after query analysis, before planning.
If premise is incorrect, provides a correction and adjusted query.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PREMISE_VALIDATOR_PROMPT = """You are a premise validation agent. Your job is to check whether the user's question contains a false, misleading, or unverified premise.

## What Is a False Premise?
A question has a false premise when it assumes something that is not true.

Examples:
- "Why is there an ongoing war between USA and Iran?" → FALSE: There is no declared war.
- "Why did Einstein fail math?" → FALSE: Einstein did not fail math.
- "Why does gradient descent zigzag?" → CORRECT: Gradient descent does exhibit zigzag behavior.
- "What is the capital of France?" → CORRECT: No false premise.

## Your Job
1. Identify the key premises/assumptions embedded in the question.
2. Evaluate each premise for factual accuracy using your knowledge.
3. If ANY premise is false or misleading, flag it and provide a correction.
4. If all premises are correct, confirm and pass through.

## Output JSON Schema
{
  "premise_status": "<correct|incorrect|uncertain>",
  "premises_found": ["<premise 1>", "<premise 2>"],
  "issues": ["<description of the false premise, if any>"],
  "correction": "<factual correction if premise is false, empty string if correct>",
  "corrected_query": "<rewritten query that removes the false premise, or original query if correct>"
}

## Rules
- Be conservative: only flag premises you are CONFIDENT are false.
- For uncertain or debatable premises, use "uncertain" status.
- The corrected_query should still address what the user likely wants to know.
- Output valid JSON only.
"""


def validate_premise(analysis: dict) -> dict:
    """
    Check whether the query contains a false premise.

    Args:
        analysis: The query analysis dict from query_analyzer

    Returns:
        dict with premise_status, correction, corrected_query
    """
    query = analysis.get("original_query", analysis.get("refined_query", ""))

    user_message = f"""Analyze the premises in this question:

Question: {query}

Query Analysis:
{json.dumps(analysis, indent=2)}"""

    try:
        from utils.llm_client import get_json_response
        result = get_json_response(
            messages=[
                {"role": "system", "content": PREMISE_VALIDATOR_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=512
        )

        # Ensure required fields
        result.setdefault("premise_status", "correct")
        result.setdefault("correction", "")
        result.setdefault("corrected_query", query)
        result.setdefault("premises_found", [])
        result.setdefault("issues", [])

        return result

    except json.JSONDecodeError as e:
        print(f"Error parsing premise validation JSON: {e}")
        return {"premise_status": "correct", "correction": "", "corrected_query": query}
    except Exception as e:
        print(f"Error in premise validation: {e}")
        return {"premise_status": "correct", "correction": "", "corrected_query": query}
