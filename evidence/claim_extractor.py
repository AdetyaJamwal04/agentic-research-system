"""
Claim Extractor — Converts retrieved passages into atomic, structured claims.

Each claim has: claim text, mechanism, source_id, confidence.
This replaces document-level evidence with claim-level evidence
for more precise reasoning, citation, and contradiction detection.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CLAIM_EXTRACTOR_PROMPT = """You are a claim extraction agent. Convert evidence passages into atomic, structured claims.

## Your Job
For each evidence passage, extract individual factual claims. Each claim must be:
- **Atomic**: one specific fact or mechanism per claim
- **Self-contained**: understandable without reading the original passage
- **Precise**: include the specific mechanism, cause, or relationship
- **Grounded**: only extract what the passage actually states

## Input
You receive numbered evidence passages with source IDs like [1], [2], etc.

## Output JSON Schema
{
  "claims": [
    {
      "claim": "<atomic factual statement>",
      "mechanism": "<the causal mechanism or relationship described>",
      "source_id": <integer source ID from the passage label>,
      "confidence": <0.0-1.0 how explicitly the passage supports this claim>
    }
  ]
}

## Confidence Scale
- 0.9-1.0: Passage directly and explicitly states this claim
- 0.7-0.8: Passage strongly implies this claim
- 0.5-0.6: Passage provides partial evidence for this claim
- Below 0.5: Do not extract — too speculative

## Rules
- Extract 2-5 claims per passage. Prefer fewer, precise claims over many vague ones.
- Do NOT infer claims the passage doesn't support.
- Do NOT combine information from different passages into one claim.
- Keep source_id matching the [N] label of the passage the claim came from.
- Output valid JSON only.
"""


def extract_claims(query: str, numbered_evidence: str) -> list:
    """
    Extract atomic claims from numbered evidence passages.

    Args:
        query: The user's original question (provides context)
        numbered_evidence: Formatted evidence with [1], [2] labels

    Returns:
        List of claim dicts: [{claim, mechanism, source_id, confidence}, ...]
    """
    user_message = f"""Question: {query}

Evidence:
{numbered_evidence}"""

    try:
        from utils.llm_client import get_json_response
        result = get_json_response(
            messages=[
                {"role": "system", "content": CLAIM_EXTRACTOR_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        claims = result.get("claims", [])

        # Validate and clean claims
        valid_claims = []
        for c in claims:
            if isinstance(c, dict) and "claim" in c:
                valid_claims.append({
                    "claim": c.get("claim", ""),
                    "mechanism": c.get("mechanism", ""),
                    "source_id": c.get("source_id", 0),
                    "confidence": min(1.0, max(0.0, float(c.get("confidence", 0.5)))),
                })
        return valid_claims

    except json.JSONDecodeError as e:
        print(f"Error parsing claims JSON: {e}")
        return []
    except Exception as e:
        print(f"Error extracting claims: {e}")
        return []


def format_claims_for_display(claims: list) -> str:
    """Format claims for console output."""
    lines = []
    for i, c in enumerate(claims, 1):
        conf = f"{c['confidence']:.0%}"
        lines.append(f"  [{c['source_id']}] ({conf}) {c['claim']}")
        if c.get("mechanism"):
            lines.append(f"       mechanism: {c['mechanism']}")
    return "\n".join(lines)
