"""
Context Compressor — Extracts relevant passages from documents.
Routes through the centralized LLM client for failover support.
"""

from utils.llm_client import get_llm_response

COMPRESSOR_SYSTEM_PROMPT = """You are an evidence extraction assistant. Extract passages from the document that help answer the user's question.

## Rules
- Extract relevant sentences or short passages EXACTLY as written.
- Do NOT summarize, paraphrase, or rewrite.
- Preserve technical explanations and important details.
- Ignore navigation text, headers, and unrelated sections.
- Prefer passages that define concepts, explain mechanisms, or describe relevant relationships.

## Output Format
PASSAGES:
1. <verbatim passage>
2. <verbatim passage>
3. <verbatim passage>
"""


def compress_context(query, documents):

    compressed_context = []

    for doc in documents:

        user_message = f"""Question:
{query}

Document:
{doc}"""

        try:
            result = get_llm_response(
                messages=[
                    {"role": "system", "content": COMPRESSOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            compressed_context.append(result.strip())
        except Exception as e:
            print(f"Error compressing document, using raw content: {e}")
            compressed_context.append(doc[:800])

    return compressed_context
