# Agentic Research System

An AI-powered research agent that autonomously searches the web, retrieves local documents, extracts claims, validates evidence, and produces cited answers across multi-turn conversations.

## 🎯 Features

- **Multi-Agent Pipeline**: 12-step intelligent research workflow
  - Query analysis & premise validation
  - Adaptive planning with dynamic task generation
  - Dual-mode retrieval (web + local documents)
  - Evidence synthesis & claim extraction
  - Reasoning with iterative critic loop

- **Hybrid Retrieval System**
  - Web search via Tavily API
  - Vector embeddings (ChromaDB)
  - BM25 sparse search
  - Reciprocal Rank Fusion fusion

- **Session Management**
  - Multi-turn conversations with persistent history
  - SQLite-backed session storage
  - Automatic session titling

- **Long-Term Memory**
  - Semantic deduplication of verified claims
  - Quality-gated knowledge graph
  - Cross-session claim retrieval

- **Fallback LLM Integration**
  - Primary: Groq (llama-3.3-70b-versatile)
  - Fallback: Google Gemini (gemini-2.0-flash)
  - Automatic failover with exponential backoff

- **Dual Interface**
  - Modern web UI with live streaming pipeline
  - Interactive CLI with history & memory management
  - RESTful API with WebSocket support

- **Document Management**
  - PDF upload & ingestion
  - Session-scoped document storage
  - Automatic chunking & embedding

## 🏗️ Architecture

```
User Query
    ↓
Session Manager (loads conversation context)
    ↓
Query Analyzer (intent, entities, sub-questions)
    ↓
Premise Validator (factual premise checking)
    ↓
Planner (multi-task research plan)
    ↓
Query Generator (task-specific search queries)
    ↓
Retriever Agent (Tavily + ChromaDB + BM25)
    ↓
Domain Filter → Evidence Pool → Reranker → Context Compressor
    ↓
Claim Extractor → Evidence Synthesizer
    ↓
Reasoning Agent + Critic Loop (max 3 iterations)
    ↓
Persist: History (SQLite) + LTM (ChromaDB)
```

### Project Structure

```
agentic-research-system/
├── agents/                 # AI agents (query analysis, planning, reasoning)
├── retrieval/             # Retrieval modules (vector, BM25, reranking)
├── memory/                # Session, history, evidence, long-term memory
├── evidence/              # Claim extraction & analysis
├── pipelines/             # Research pipeline orchestration
├── planning/              # Dynamic task planning
├── tools/                 # Web search, file loading
├── utils/                 # LLM client, logging, token counting
├── evaluation/            # Benchmarking & quality checking
├── templates/             # Frontend HTML/CSS/JS
├── data/                  # SQLite DB, ChromaDB, documents
├── tests/                 # Unit tests
├── app.py                 # FastAPI server
├── main.py                # CLI interface
├── config.py              # Configuration & settings
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or poetry

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd agentic-research-system
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_key
# TAVILY_API_KEY=your_tavily_key
# GEMINI_API_KEY=your_gemini_key (optional, for fallback)
```

### Usage

#### **Web UI** (Recommended)

```bash
python -m uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000` in your browser to access the chat interface.

**Features:**
- Create & manage sessions
- Upload PDFs for research
- View live pipeline progress
- Expand sources & claims metadata
- Toggle "My Docs" for local-only research

#### **CLI Interface**

```bash
python main.py
```

**Commands:**
- `your query` — Run research on the query
- `upload path/to/file.pdf` — Ingest a PDF
- `history` — View past queries
- `ltm` — View long-term memory
- `clear` — Reset all memory

Example:
```
> Why does gradient descent zigzag?
[12-step pipeline runs]
> upload myresearch.pdf
> history
> exit
```

## 📡 API Endpoints

### Sessions
- `POST /api/sessions` — Create new session
- `GET /api/sessions` — List all sessions
- `GET /api/sessions/{id}` — Get session + messages
- `PUT /api/sessions/{id}` — Rename session
- `DELETE /api/sessions/{id}` — Delete session

### Research
- `WS /ws/research` — WebSocket for streaming research
- `POST /api/research` — Stateless research (legacy)

### Documents
- `POST /api/sessions/{id}/upload` — Upload PDF to session
- `POST /api/upload` — Upload PDF (stateless, legacy)

### Memory & History
- `GET /api/history` — Research history
- `GET /api/ltm` — Long-term memory contents
- `DELETE /api/history` — Clear research history
- `DELETE /api/ltm` — Clear long-term memory

### WebSocket Message Format

**Request:**
```json
{
  "query": "Your research question",
  "session_id": "uuid",
  "max_iterations": 3,
  "use_memory": true
}
```

**Response Stream (JSON events):**
```json
{"type": "step", "step": "Query Analysis", "status": "in_progress"}
{"type": "step", "step": "Query Analysis", "status": "complete", "duration": 1.2}
{"type": "complete", "answer": "...", "sources": [...], "claims": [...]}
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model Settings
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # Groq model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # Sentence transformer

# Retrieval Settings
MAX_QUERIES_PER_TASK = 6              # Search queries per task
MAX_RETRIEVED_DOCS = 20               # Initial retrieval count
RERANK_TOP_K = 3                      # Documents after reranking
CHUNK_SIZE = 500                      # PDF chunk size (words)
CHUNK_OVERLAP = 50                    # Overlap between chunks

# Critic Settings
CRITIC_THRESHOLD = 8                  # Score for "sufficient" answer
MAX_RETRY_ITERATIONS = 3              # Reasoning loop iterations
```

## 🔑 API Keys Required

1. **Groq** (Primary LLM)
   - Free tier: 6,000 calls/month
   - Sign up: https://console.groq.com

2. **Tavily** (Web Search)
   - Free tier: 1,000 calls/month
   - Sign up: https://tavily.com

3. **Google Gemini** (Optional Fallback)
   - Free tier: 60 requests/minute
   - Sign up: https://aistudio.google.com

## 📦 Dependencies

### Core
- **FastAPI** — Web framework
- **Uvicorn** — ASGI server
- **LangChain** — LLM orchestration
- **LangGraph** — Agentic planning (scaffolded)

### Data & Embeddings
- **ChromaDB** — Vector database
- **Sentence Transformers** — Embeddings
- **Rank-BM25** — Sparse search
- **PyPDF2** — PDF processing
- **Scikit-learn** — ML utilities

### LLM & APIs
- **Groq** — Primary LLM
- **Google Generative AI** — Gemini fallback
- **Tavily Python** — Web search

### Other
- **python-dotenv** — Environment variables
- **qdrant-client** — Vector DB client

## 🗄️ Data Storage

All data stored locally in `data/` directory:

```
data/
├── research.db              # SQLite (sessions, messages, history)
├── chroma_db/              # ChromaDB (document embeddings)
├── ltm_chroma_db/          # ChromaDB (long-term memory)
├── documents/              # Uploaded PDFs
└── bm25_index.pkl          # BM25 index (optional)
```

**Reset everything:**
```bash
rm -rf data/
```

## 🚀 Deployment

### Free Tier Options

#### **Render (Recommended)**
1. Push to GitHub
2. Create Web Service on [render.com](https://render.com)
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app:app --host 0.0.0.0 --port 8000`
5. Add environment variables
6. Deploy

#### **Railway**
- $5/month free credit
- Sign up at [railway.app](https://railway.app)
- Connect GitHub repo
- Auto-detects requirements.txt

#### **PythonAnywhere**
- 100MB free disk
- Upload project to [pythonanywhere.com](https://pythonanywhere.com)
- Configure WSGI with Gunicorn

#### **Fly.io**
- $5/month free credit
- Requires Dockerfile (simple setup)

#### **LocalTunnel** (Your Machine)
```bash
npm install -g localtunnel
localtunnel --port 8000 --subdomain my-research-system
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides.

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## 📊 Pipeline Steps Explained

1. **Memory Load** — Retrieves relevant LTM claims & conversation history
2. **Query Analysis** — Extracts intent, entities, sub-questions
3. **Premise Validation** — Checks factual assumptions in query
4. **Planning** — Creates multi-task research plan
5. **Query Generation** — Generates search queries per task
6. **Retrieval** — Web search (Tavily) + local retrieval (Chrome DB + BM25)
7. **Domain Filter** — LLM removes off-topic documents
8. **Evidence Pool** — Deduplicates and tags documents
9. **Reranking** — LLM scores documents by relevance
10. **Context Compression** — Extracts key passages for reasoning
11. **Claim Extraction** — Identifies claims with confidence scores
12. **Evidence Synthesis** → Identifies themes, conflicts, gaps
13. **Reasoning + Critic Loop** — Generates answer, evaluates quality, optionally re-retrieves
14. **Persistence** — Saves to history (SQLite) & LTM (ChromaDB)

## 🔄 Critic Loop

If answer quality score < 8:
1. Critic identifies missing aspects
2. Dynamic planner generates new tasks
3. Retriever fetches additional evidence
4. Reasoning agent generates improved answer
5. Repeat up to MAX_RETRY_ITERATIONS (default: 3)

## 💾 Environment Variables

```bash
# Required
GROQ_API_KEY=sk-...                  # Groq API key

# Recommended
TAVILY_API_KEY=tv...                 # Tavily API key
GEMINI_API_KEY=AIzaSy...            # Google Gemini API key (fallback)

# Optional
DEBUG=false                          # Enable debug logging
```

## 🎓 Example Usage

**Terminal Session:**
```bash
$ python main.py
> What are the latest advances in quantum computing?
[Pipeline runs...]
Answer: [Long-form research with citations]
Sources: [arXiv, Nature, etc.]

> upload quantum_paper.pdf
[PDF ingested into ChromaDB]

> Are there any quantum advantages in cryptography?
[Uses both web + local document search]

> history
[Shows all past queries]

> ltm
[Shows verified claims in long-term memory]

> exit
```

**API Example (cURL):**
```bash
# Create session
curl -X POST http://localhost:8000/api/sessions

# Upload PDF
curl -X POST http://localhost:8000/api/sessions/{id}/upload \
  -F "file=@myresearch.pdf"

# Stream research via WebSocket
wscat -c ws://localhost:8000/ws/research
> {"query": "Your question", "session_id": "...", "max_iterations": 3}
```

## 📝 Notes

- All conversions are async-safe via thread pool executors
- ChromaDB uses cosine similarity (L2 normalized dense embeddings)
- BM25 index is optional; hybrid search gracefully falls back to vector-only
- Long-term memory deduplicates with threshold < 0.05 cosine distance
- Groq fallback to Gemini activates on cooldown (429) or network errors

## 🛠️ Troubleshooting

### Issue: "GROQ_API_KEY not set"
```bash
# Check .env exists
ls -la .env
# Ensure .env is in project root, not in venv/
```

### Issue: ChromaDB connection error
```bash
# Reset database
rm -rf data/chroma_db
# Will be auto-recreated on next run
```

### Issue: Tavily API quota exceeded
```bash
# Switch to local_only mode in settings
# Or use fallback to Gemini for less stricter web search
```

### Issue: Out of memory on shared hosting
```bash
# Reduce MAX_RETRIEVED_DOCS in config.py
# Reduce CHUNK_SIZE for PDFs
# Use local_only mode to skip web search
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit with clear messages
4. Push to branch
5. Open a Pull Request

## 📄 License

MIT License — See LICENSE file for details

## 🙏 Acknowledgments

- **Groq** for high-performance LLM inference
- **Tavily** for web search API
- **Google** for Gemini fallback
- **LangChain** for orchestration framework
- **ChromaDB** for vector database

## 📧 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your-email@example.com

---

**Built with ❤️ using LangChain, FastAPI, and ChromaDB**

Last Updated: March 18, 2026
