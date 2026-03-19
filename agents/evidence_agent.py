"""
Evidence Agent — Merges, deduplicates, and organizes evidence passages
into a structured brief for the reasoning agent.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYNTHESIZER_SYSTEM_PROMPT = """You are an evidence synthesis agent. You receive multiple evidence passages and organize them into a structured brief.

## Your Job
1. Group related evidence by theme or mechanism.
2. Merge overlapping passages — remove redundancy, keep the strongest version.
3. Flag conflicting claims between passages.
4. Identify coverage gaps — what aspects of the question lack evidence.

## Output JSON Schema
{
  "evidence_brief": [
    {
      "theme": "<mechanism or concept name>",
      "points": ["<key finding 1>", "<key finding 2>"],
      "sources": [<source_id_1>, <source_id_2>]
    }
  ],
  "conflicts": ["<description of conflicting claims, if any>"],
  "coverage_gaps": ["<aspect that lacks evidence>"]
}

## Rules
- Each theme should represent one distinct mechanism or concept.
- Points should be concise factual statements, not full paragraphs.
- Keep source IDs (the [N] numbers) to maintain citation traceability.
- Output valid JSON only.
"""


def synthesize_evidence(query: str, numbered_evidence: str) -> dict:
    """
    Takes numbered evidence passages and produces a structured evidence brief.
    """
    user_message = f"""Question: {query}

Evidence:
{numbered_evidence}"""

    try:
        from utils.llm_client import get_json_response
        return get_json_response(
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=1500
        )
    except json.JSONDecodeError as e:
        print(f"Error parsing synthesis JSON: {e}")
        return {"evidence_brief": [], "conflicts": [], "coverage_gaps": ["synthesis failed"]}
    except Exception as e:
        print(f"Error in evidence synthesis: {e}")
        return {"evidence_brief": [], "conflicts": [], "coverage_gaps": [str(e)]}


def format_synthesis_for_reasoning(synthesis: dict) -> str:
    """
    Convert the structured evidence brief into a readable format
    for the reasoning agent.
    """
    parts = []
    
    for item in synthesis.get("evidence_brief", []):
        theme = item.get("theme", "Unknown")
        sources = item.get("sources", [])
        source_str = ", ".join(str(s) for s in sources)
        parts.append(f"## {theme} [sources: {source_str}]")
        for point in item.get("points", []):
            parts.append(f"- {point}")
        parts.append("")

    conflicts = synthesis.get("conflicts", [])
    if conflicts:
        parts.append("## Conflicting Claims")
        for c in conflicts:
            parts.append(f"- {c}")
        parts.append("")

    gaps = synthesis.get("coverage_gaps", [])
    if gaps:
        parts.append("## Coverage Gaps")
        for g in gaps:
            parts.append(f"- {g}")

    return "\n".join(parts)
