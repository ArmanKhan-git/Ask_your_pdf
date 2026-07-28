# 📄 Chat With Your PDF — Agentic RAG

An agentic Retrieval-Augmented Generation (RAG) app that lets you upload any PDF and have a grounded, cited conversation with it. Built with **LangGraph** for control flow, **hybrid retrieval + reranking** for accurate context, and **Streamlit** for the interface — all deployable in a single script.

Unlike a naive "retrieve-then-generate" chatbot, this system actively checks its own retrieval quality before answering, and refuses to answer when the document genuinely doesn't contain the information — rather than guessing.

---
## 🚀 [Live Demo](https://askyourpdfqna.streamlit.app/)
---

## ✨ Features

- **Upload any PDF, chat immediately** — no pre-indexing step, no separate ingestion script. Drop a file in and start asking questions.
- **Automatic re-indexing on new upload** — upload a different PDF mid-session and the app detects it and rebuilds automatically, no manual reset needed.
- **Hybrid retrieval** — combines dense vector search (FAISS) with keyword search (BM25) via a weighted ensemble, so both semantic similarity *and* exact-term matches are captured.
- **Reranking** — retrieved candidates are re-scored by a Cohere cross-encoder reranker before being passed to the LLM, improving precision beyond what vector similarity alone provides.
- **Agentic, self-correcting flow** — a LangGraph state machine that:
  - Rewrites follow-up questions into standalone queries using conversation history (so "what about dosage?" correctly resolves to what it's actually asking about)
  - Grades whether retrieved context is actually sufficient to answer the question
  - Falls back to an honest "I don't know" instead of hallucinating when context is insufficient
- **Streaming responses** — answers stream token-by-token in the UI instead of appearing all at once.
- **Multi-turn memory** — conversation history persists within a session via LangGraph's checkpointing.
- **Source citations** — every answer includes the page number(s) it was grounded in.
- **Observability** — LangSmith tracing integrated for inspecting every node execution, retry, and LLM call.

---

## 🧠 How it works

```
User question
     │
     ▼
┌─────────────┐
│   rewrite    │  → resolves pronouns/context using chat history,
└─────────────┘     produces a standalone search query
     │
     ▼
┌─────────────┐
│  retrieve    │  → hybrid search (FAISS + BM25) → Cohere rerank
└─────────────┘     returns the most relevant chunks
     │
     ▼
┌─────────────┐
│    grade     │  → LLM judges: does this context actually
└─────────────┘     answer the question?
     │
     ├── insufficient ──► no_answer ──► END
     │
     └── sufficient
             │
             ▼
      ┌─────────────┐
      │   generate   │  → answer grounded strictly in retrieved
      └─────────────┘     context, with page citations
             │
             ▼
            END
```

Each box above is a **node** in a LangGraph `StateGraph` — a shared state object (question, retrieved documents, chat history, answer, citations) flows through and is updated by each node, with conditional routing based on the grading step.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | Google Gemini (`gemini-3.5-flash`) |
| Embeddings | HuggingFace (`BAAI/bge-base-en-v1.5`, local) |
| Dense retrieval | FAISS |
| Keyword retrieval | BM25 |
| Reranking | Cohere Rerank (`rerank-english-v3.0`) |
| Memory / persistence | LangGraph `MemorySaver` (in-session) |
| Observability | LangSmith |
| UI | Streamlit |

> **Note:** the embedding model (`BAAI/bge-base-en-v1.5`) runs **locally** via `sentence-transformers`, not through an API. This avoids embedding-quota/rate-limit issues entirely, but means the **first document upload will be slower** — the model has to download and load into memory on first use (cached afterward via `@st.cache_resource`), and embedding itself runs on CPU rather than a remote server. Larger PDFs will take proportionally longer to process. This is a deliberate tradeoff: no external rate limits, at the cost of local compute time.

---

## 🚀 Getting Started

### 1. Clone and install
```bash
git clone https://github.com/ArmanKhan-git/Ask_your_pdf.git
cd Ask_your_pdf
python -m venv venv
source venv\Scripts\activate   
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=chat-with-your-pdf
```

### 3. Run locally
```bash
streamlit run app.py
```

The app opens in your browser. Upload a PDF and start asking questions.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select this repo and `app.py` as the entry point.
4. Under **Advanced settings → Secrets**, add the same key-value pairs from your `.env` file (Streamlit Cloud doesn't read local `.env` files — secrets are managed through their dashboard instead).
5. Deploy.

---

## 📌 Design Notes

- **Why hybrid retrieval instead of just dense vectors?** Vector similarity search is good at capturing semantic meaning but can miss exact keyword matches (e.g. specific drug names, codes, proper nouns). BM25 catches those. Combining both with a weighted ensemble covers more retrieval failure modes than either alone.
- **Why rerank on top of hybrid retrieval?** Vector/BM25 search is optimized for speed over large corpora, not precision. A cross-encoder reranker looks at the query and each candidate together and re-scores them — more accurate, and cheap to run over just the top-k candidates rather than the whole index.
- **Why grade relevance before generating?** Retrieval always returns *something*, even when nothing in the document actually answers the question. Without an explicit relevance check, the LLM will often generate a plausible-sounding but ungrounded answer instead of admitting it doesn't know — a dangerous failure mode when the app is meant to answer strictly from source material.
- **Why a local embedding model instead of an API-based one?** API-based embedding providers (e.g. Gemini's embedding endpoint) are fast and simple but subject to rate limits/quotas, which can interrupt ingestion partway through a large document. Running embeddings locally via `sentence-transformers` removes that dependency entirely — at the cost of a slower cold start (first-time model download/load) and CPU-bound processing time instead of near-instant API calls.
- **Why in-memory checkpointing instead of a persistent database?** This app is intentionally scoped to single-session use (one document, one conversation, no cross-session history). `MemorySaver` provides full multi-turn memory within a session without the operational overhead of a database — the right tradeoff for this project's scope.


