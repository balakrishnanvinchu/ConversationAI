import json, pickle
from rank_bm25 import BM25Okapi

with open("data/corpus_chunks.json") as f:
    data = json.load(f)

corpus = [d["text"].split() for d in data]

bm25 = BM25Okapi(corpus)

with open("data/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("BM25 index created")
