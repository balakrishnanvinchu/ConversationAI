import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rrf import rrf_fusion
from src.generator import generate_answer


# ---------------- LOAD MODELS ----------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

# ---------------- LOAD DATA ----------------

with open("data/corpus_chunks.json") as f:
    corpus = json.load(f)

texts = [c["text"] for c in corpus]

# Load FAISS index
index = faiss.read_index("data/faiss.index")

if not index.is_trained:
    raise RuntimeError("FAISS index not trained")

# Load BM25
with open("data/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)


# ---------------- RAG PIPELINE ----------------

def run_rag(query, top_k=10, final_k=5):

    if not query.strip():
        raise ValueError("Empty query")

    # ---------- Dense Retrieval ----------

    q_emb = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    dense_scores, dense_ids = index.search(q_emb, top_k)

    dense_results = []
    for rank, (idx, score) in enumerate(zip(dense_ids[0], dense_scores[0])):
        dense_results.append({
            "rank": rank + 1,
            "chunk": texts[idx],
            "url": corpus[idx]["url"],
            "score": float(score)
        })

    # ---------- Sparse Retrieval ----------

    scores = bm25.get_scores(query.split())

    sparse_top = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    sparse_results = []
    for rank, (idx, score) in enumerate(sparse_top):
        sparse_results.append({
            "rank": rank + 1,
            "chunk": texts[idx],
            "url": corpus[idx]["url"],
            "score": float(score)
        })

    # ---------- RRF Fusion ----------

    fused = rrf_fusion(dense_ids[0], sparse_top)

    rrf_results = []
    for idx, score in fused:
        rrf_results.append({
            "chunk": texts[idx],
            "url": corpus[idx]["url"],
            "rrf_score": float(score)
        })

    # ---------- Final Context ----------

    contexts = [item["chunk"] for item in rrf_results[:final_k]]
    sources = [item["url"] for item in rrf_results[:final_k]]

    # ---------- Answer Generation ----------

    answer = generate_answer(query, contexts)

    # ---------- Return ----------

    return {
        "answer": answer,
        "sources": sources,
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "rrf_results": rrf_results
    }
