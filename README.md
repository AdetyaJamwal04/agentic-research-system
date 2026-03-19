# 🔬 Agentic Research System

An AI-powered multi-agent research system that takes a user query, searches the web and local documents, extracts claims, validates evidence, and produces **cited, fact-checked answers** — all through a ChatGPT-style conversational interface.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-yellow)

---

## ✨ Features

- **12-step agentic pipeline** — query analysis → premise validation → planning → retrieval → filtering → reranking → reasoning → evidence extraction → citation mapping → critique → iterative refinement → final answer
- **Multi-turn chat sessions** — persistent conversations with context, rename, delete
- **Hybrid retrieval** — web search (Tavily) + local vector search (ChromaDB) + BM25
- **PDF upload & analysis** — upload documents and query them with the "My Docs" toggle
- **Claim extraction & validation** — extracts factual claims with confidence scores
- **Citation-backed answers** — every claim is traceable to its source
- **Self-critique loop** — the system evaluates its own answers and iterates up to 3 times
- **LLM failover** — Groq (primary) → Gemini (fallback) with retry + exponential backoff
- **Long-term memory** — verified claims persist across sessions via ChromaDB

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           12-Step Pipeline              │
│                                         │
│  1. Query Analysis (intent, entities)   │
│  2. Premise Validation                  │
│  3. Research Planning                   │
│  4. Retrieval (Web + Local + BM25)      │
│  5. Domain Relevance Filtering          │
│  6. Reranking (Cross-Encoder)           │
│  7. Reasoning & Synthesis              │
│  8. Evidence Extraction                 │
│  9. Citation Mapping                    │
│ 10. Critic Evaluation                  │
│ 11. Iterative Refinement (if needed)   │
│ 12. Final Answer                       │
└─────────────────────────────────────────┘
    │
    ▼
Cited Answer + Sources + Claims + Confidence
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Groq API Key](https://console.groq.com) (free)
- [Tavily API Key](https://tavily.com) (free tier)
- [Gemini API Key](https://aistudio.google.com/apikey) (optional fallback)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/agentic-research-system.git
cd agentic-research-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Run

```bash
# Web UI (recommended)
uvicorn app:app --reload --port 8000
# Open http://localhost:8000

# CLI mode
python main.py
```

## 📁 Project Structure

```
├── agents/                  # LLM-powered agents
│   ├── query_analyzer.py    # Query intent & entity extraction
│   ├── premise_validator.py # Validates query assumptions
│   ├── planner.py           # Creates research plan
│   ├── query_generator.py   # Generates search queries per task
│   ├── retriever_agent.py   # Hybrid web + local retrieval
│   ├── relevance_filter.py  # Domain relevance filtering
│   ├── reasoning_agent.py   # Synthesizes answer from evidence
│   ├── evidence_agent.py    # Extracts structured evidence
│   ├── critic_agent.py      # Self-critique & scoring
│   └── report_agent.py      # Report generation (unused)
│
├── retrieval/               # Retrieval & ranking
│   ├── hybrid_retriever.py  # ChromaDB + BM25 fusion
│   ├── reranker.py          # Cross-encoder reranking
│   ├── context_compressor.py# Compresses context to fit LLM window
│   ├── vector_retriever.py  # ChromaDB vector search
│   └── web_retriever.py     # Web content fetching
│
├── memory/                  # Persistence layer
│   ├── database.py          # SQLite schema & queries
│   ├── session_manager.py   # Session CRUD & messages
│   ├── long_term_memory.py  # Verified claims (ChromaDB)
│   ├── evidence_pool.py     # Evidence aggregation
│   └── research_history.py  # Past research tracking
│
├── pipelines/
│   └── research_pipeline.py # Orchestrates the 12-step pipeline
│
├── evidence/
│   └── claim_extractor.py   # Extracts claims with confidence
│
├── tools/
│   ├── web_search.py        # Tavily API integration
│   └── file_loader.py       # PDF parsing & chunking
│
├── utils/
│   └── llm_client.py        # Groq + Gemini failover client
│
├── templates/
│   └── index.html           # Chat UI (single-page app)
│
├── app.py                   # FastAPI server + WebSocket
├── main.py                  # CLI entry point
├── config.py                # Centralized configuration
└── system_architecture.txt  # Detailed architecture docs
```

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Primary LLM provider (Llama 3.3 70B) |
| `TAVILY_API_KEY` | ✅ | Web search API |
| `GEMINI_API_KEY` | Optional | Fallback LLM (Gemini 2.0 Flash) |

## 🐳 Docker Deployment

```bash
docker build -t agentic-research .
docker run -p 8000:8000 --env-file .env agentic-research
```

## 📄 License

MIT
