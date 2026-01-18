import json, pickle, faiss
from sentence_transformers import SentenceTransformer
from rrf import rrf_fusion
from generator import generate_answer

model = SentenceTransformer("all-mpnet-base-v2")

with open("data/corpus_chunks.json") as f:
    corpus = json.load(f)

texts = [c["text"] for c in corpus]

index = faiss.read_index("data/faiss.index")

with open("data/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)


def run_rag(query, top_k=10, final_k=5):

    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)

    _, dense_ids = index.search(q_emb, top_k)

    scores = bm25.get_scores(query.split())
    sparse = sorted(enumerate(scores),
                    key=lambda x:x[1],
                    reverse=True)[:top_k]

    fused = rrf_fusion(dense_ids[0], sparse)

    contexts = [texts[i] for i,_ in fused[:final_k]]
    urls = [corpus[i]["url"] for i,_ in fused]

    answer = generate_answer(query, contexts)

    return answer, urls
