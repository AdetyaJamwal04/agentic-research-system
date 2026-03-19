from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key= os.getenv("GROQ_API_KEY")
)

RELEVANCE_SYSTEM_PROMPT = """You are a document relevance classifier for a retrieval-augmented generation system.

Classify whether the document contains information useful for answering the user's question.

## Classification Rules
- RELEVANT: The document explains concepts, mechanisms, definitions, or facts that contribute to answering the question. Partial or indirect relevance counts.
- IRRELEVANT: The document is off-topic, contains only boilerplate/navigation text, or has no meaningful connection to the question.

Respond with exactly one word: RELEVANT or IRRELEVANT"""

def filter_documents(query, documents):
    
    filtered_docs = []

    for doc in documents[:10]:

        user_message = f"""Question: {query}

Document: {doc}"""

        try:
            response = client.chat.completions.create(
                model= "llama-3.3-70b-versatile",
                messages= [
                    {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                max_tokens=16,
            )

            answer = response.choices[0].message.content.strip()

            if answer.strip() == "RELEVANT":
                filtered_docs.append(doc)

        except Exception as e:
            print(f"Error filtering document, skipping: {e}")
            continue

    return filtered_docs