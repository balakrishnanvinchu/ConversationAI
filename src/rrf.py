from collections import defaultdict

def rrf_fusion(dense_ids, sparse_ids, k=60):
    scores = defaultdict(float)

    for rank, d in enumerate(dense_ids):
        scores[d] += 1/(k + rank + 1)

    for rank, (d, _) in enumerate(sparse_ids):
        scores[d] += 1/(k + rank + 1)

    return sorted(scores.items(), key=lambda x:x[1], reverse=True)
