# Document Intelligence Starter

A small Retrieval-Augmented Generation (RAG) application for practicing GenAI engineering.

## Features
- Upload PDF, DOCX, TXT, and Markdown files
- Extract text and split it into overlapping chunks
- Create embeddings with `text-embedding-3-small`
- Store vectors in a local in-memory FAISS index
- Retrieve the most relevant chunks for a question
- Generate a grounded answer with source/chunk citations
- Inspect retrieved evidence in the UI

## Requirements
- Python 3.10+
- An OpenAI API key with API access/billing enabled

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Paste your API key in the sidebar. Alternatively set `OPENAI_API_KEY` in your environment.

## Architecture

```text
PDF / DOCX / TXT
      |
      v
Text extraction
      |
      v
Chunking (900 chars, 150 overlap)
      |
      v
Embedding model
      |
      v
FAISS vector index
      |
User question -> question embedding
      |                 |
      +---- similarity search
                 |
          Top 4 chunks
                 |
          LLM with context
                 |
        Answer + citations
```

## Important limitations
- The FAISS index exists only in memory and disappears when the app restarts.
- This starter uses character-based chunking, not token-aware chunking.
- PDF extraction works best on text PDFs; scanned PDFs need OCR.
- Similarity scores are retrieval signals, not calibrated confidence scores.
- This is a learning demo, not a production-ready secure document platform.
- Do not upload confidential or regulated documents to an API unless you have approval.

## Suggested interview practice roadmap

1. Explain the RAG pipeline: ingestion, chunking, embeddings, retrieval, generation.
2. Change chunk size and overlap; compare retrieval quality.
3. Add metadata such as page number and document ID.
4. Add OCR for scanned PDFs.
5. Add persistent storage (Chroma or a saved FAISS index).
6. Add hybrid retrieval (keyword + vector search) and reranking.
7. Add evaluation: Recall@K, answer faithfulness, and a small question/ground-truth dataset.
8. Add FastAPI endpoints and Docker deployment.
9. Add authentication, tenant isolation, file validation, logging, and deletion controls before production use.

## Core interview questions
- Why do we chunk documents?
- What are embeddings and cosine similarity?
- Why normalize vectors when using inner product as cosine similarity?
- What happens when relevant information is split across chunk boundaries?
- How do you reduce hallucinations?
- How would you evaluate retrieval separately from answer quality?
- When would you use OCR, hybrid search, or a reranker?
- What security controls are needed for multi-user document Q&A?
