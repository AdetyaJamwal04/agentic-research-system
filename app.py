"""
FastAPI App — Exposes the research pipeline via REST API + WebSocket.
Includes session management for multi-turn conversations.

Run: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import shutil

from pipelines.research_pipeline import run_research, run_research_stream
from memory.research_history import ResearchHistory
from memory.session_manager import SessionManager
from memory.database import get_connection

app = FastAPI(
    title="Agentic Research System",
    description="AI-powered research agent with hybrid retrieval, claim extraction, and adaptive planning",
    version="2.0.0",
)

sessions = SessionManager()


# ---- Models ----

class ResearchRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 3
    use_memory: Optional[bool] = True


# ---- Session API ----

@app.post("/api/sessions")
async def create_session():
    """Create a new research session."""
    session_id = sessions.create_session()
    return JSONResponse(content={"id": session_id, "title": "New Research"})


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions, most recent first."""
    return JSONResponse(content={"sessions": sessions.list_sessions()})


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a session with all its messages."""
    session = sessions.get_session(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return JSONResponse(content=session)


@app.put("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: Request):
    """Rename a session."""
    body = await request.json()
    title = body.get("title", "Untitled")
    sessions.rename_session(session_id, title)
    return JSONResponse(content={"status": "renamed", "title": title})


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    sessions.delete_session(session_id)
    return JSONResponse(content={"status": "deleted"})


# ---- File Upload (session-scoped) ----

@app.post("/api/sessions/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload a PDF, process it, ingest into local vector DB."""
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDFs are supported"})

    uploads_dir = os.path.join(os.path.dirname(__file__), "data", "documents")
    os.makedirs(uploads_dir, exist_ok=True)
    filepath = os.path.join(uploads_dir, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from tools.file_loader import process_pdf
    from retrieval.vector_retriever import ingest_documents

    chunks = process_pdf(filepath)
    if not chunks:
        return JSONResponse(status_code=500, content={"error": "Failed to extract text from PDF"})

    added = ingest_documents(chunks)

    # Record the upload as a system message
    sessions.add_message(
        session_id, "system",
        f"Uploaded {file.filename} ({added} chunks ingested)",
        metadata={"type": "upload", "filename": file.filename, "chunks": added}
    )

    return JSONResponse(content={"message": f"Successfully ingested {added} chunks from {file.filename}"})


# ---- Legacy upload endpoint (backward compat) ----

@app.post("/api/upload")
async def upload_file_legacy(file: UploadFile = File(...)):
    """Legacy upload endpoint — no session scope."""
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDFs are supported"})

    uploads_dir = os.path.join(os.path.dirname(__file__), "data", "documents")
    os.makedirs(uploads_dir, exist_ok=True)
    filepath = os.path.join(uploads_dir, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from tools.file_loader import process_pdf
    from retrieval.vector_retriever import ingest_documents

    chunks = process_pdf(filepath)
    if not chunks:
        return JSONResponse(status_code=500, content={"error": "Failed to extract text from PDF"})

    added = ingest_documents(chunks)
    return JSONResponse(content={"message": f"Successfully ingested {added} chunks from {file.filename}"})


# ---- WebSocket Research (session-scoped) ----

@app.websocket("/ws/research")
async def websocket_research(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query")
        session_id = data.get("session_id")
        max_iterations = data.get("max_iterations", 3)
        use_memory = data.get("use_memory", True)
        local_only = data.get("local_only", False)

        # Build conversation context from session history
        conversation_context = ""
        if session_id:
            conversation_context = sessions.build_context(session_id)
            # Save the user message
            sessions.add_message(session_id, "user", query)

            # Auto-title on first message
            msgs = sessions.get_messages(session_id)
            user_msgs = [m for m in msgs if m["role"] == "user"]
            if len(user_msgs) == 1:
                sessions.auto_title(session_id, query)

        async for update in run_research_stream(
            query, max_iterations, use_memory,
            conversation_context=conversation_context,
            local_only=local_only
        ):
            await websocket.send_json(update)

            # When pipeline completes, save the assistant response
            if update.get("type") == "complete" and session_id:
                result = update.get("result", {})
                sessions.add_message(
                    session_id, "assistant",
                    result.get("answer", ""),
                    metadata={
                        "claims": result.get("claims", []),
                        "sources": result.get("sources", []),
                        "critique": result.get("critique"),
                        "steps": result.get("steps", []),
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


# ---- Legacy REST endpoint ----

@app.post("/api/research")
async def research(req: ResearchRequest):
    """Run the full research pipeline on a query (no session)."""
    result = run_research(
        query=req.query,
        max_iterations=req.max_iterations,
        use_memory=req.use_memory,
    )
    return JSONResponse(content=result)


# ---- History & Memory ----

@app.get("/api/history")
async def get_history():
    """Get research history."""
    history = ResearchHistory()
    return JSONResponse(content={"entries": history.get_history()})


@app.delete("/api/history")
async def clear_history():
    """Clear research history."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM research_history")
        conn.commit()
    finally:
        conn.close()
    return JSONResponse(content={"status": "cleared"})


@app.delete("/api/memory")
async def clear_memory():
    """Clear evidence snapshots."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM evidence_snapshots")
        conn.commit()
    finally:
        conn.close()
    return JSONResponse(content={"status": "cleared"})


@app.get("/api/ltm")
async def get_ltm():
    """Get long-term memory contents."""
    from memory.long_term_memory import LongTermMemory
    ltm = LongTermMemory()
    return JSONResponse(content={
        "claims": ltm._claims,
        "summary": ltm.summary()
    })


@app.delete("/api/ltm")
async def clear_ltm():
    """Clear long-term memory."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM claims")
        conn.commit()
    finally:
        conn.close()
    return JSONResponse(content={"status": "cleared"})


# ---- UI Route ----

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the research chat UI."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
