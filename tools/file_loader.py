import os
from PyPDF2 import PdfReader

def process_pdf(filepath: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Extracts text from a PDF file and chunks it.

    Args:
        filepath: Path to the PDF file.
        chunk_size: Number of words per chunk.
        overlap: Overlap in words between chunks.

    Returns:
        List of dictionaries containing 'content' and 'source_id' metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        reader = PdfReader(filepath)
        filename = os.path.basename(filepath)
        
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
                
        text_content = " ".join(full_text)
        words = text_content.split()
        
        chunks = []
        # Chunk text based on word count with overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source_id": filename,
                    "type": "pdf"
                }
            })
            
        return chunks
    except Exception as e:
        print(f"Error processing PDF {filepath}: {e}")
        return []
