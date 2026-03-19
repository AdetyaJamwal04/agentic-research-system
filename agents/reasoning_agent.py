"""
Reasoning Agent — Generates cited answers from evidence.
Routes through the centralized LLM client for failover support.
"""

from utils.llm_client import get_llm_response


REASONING_SYSTEM_PROMPT = """You are a technical research assistant. Answer the user's question using the provided evidence. Cite your sources with inline references.

## Reasoning Process
1. Identify ALL key mechanisms, causes, or explanations present in the evidence.
2. For EACH mechanism found, explain how it directly relates to the user's question.
3. Combine information from multiple passages to build a comprehensive explanation.
4. Use your own knowledge only to clarify or connect concepts — never contradict the evidence.
5. If the evidence contains technical terms (e.g., Hessian, condition number, curvature), explain them in context.

## Citation Rules
- Evidence passages are labeled with source numbers like [1], [2], [3].
- When you use information from a passage, cite it inline: "Gradient descent zigzags due to curvature mismatch [1][3]."
- Every factual claim in your answer MUST have at least one citation.
- You may cite multiple sources for a single claim.

## Response Guidelines
- Cover every relevant mechanism found in the evidence — do not cherry-pick only the easiest one.
- Explain WHY and HOW, not just WHAT.
- If the evidence is insufficient to fully answer, explicitly state what is missing.
- Be precise and structured.
- If prior conversation context is provided, use it to understand what the user is referring to.

## Output Format
Answer:
<comprehensive explanation with inline [N] citations>

Supporting Evidence:
- <key point from evidence with citation [N]>
- <key point from evidence with citation [N]>
"""


def generate_answer(query, evidence, conversation_context=""):
    """
    Generate a cited answer from evidence.

    Args:
        query: The user's question
        evidence: Either a pre-formatted string (with [N] labels) or a list of strings
        conversation_context: Optional prior Q&A for multi-turn support
    """
    # Handle both formats: pre-formatted evidence string or list of docs
    if isinstance(evidence, list):
        trimmed = [doc[:800] for doc in evidence[:5]]
        context = "\n\n".join(trimmed)
    else:
        context = str(evidence)

    user_message = f"""User Question:
{query}

Evidence:
{context}"""

    # Prepend conversation history if available
    if conversation_context:
        user_message = f"{conversation_context}\n\n---\n\n{user_message}"

    try:
        return get_llm_response(
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=2048,
        )
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error: Could not generate an answer. Please try again."
