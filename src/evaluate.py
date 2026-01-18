import json, pandas as pd
from rag_pipeline import run_rag

def f1(pred, truth):
    p = set(pred.lower().split())
    t = set(truth.lower().split())

    if len(p&t)==0:
        return 0

    precision = len(p&t)/len(p)
    recall = len(p&t)/len(t)

    return 2*precision*recall/(precision+recall)


with open("data/questions.json") as f:
    questions = json.load(f)

rows = []

for q in questions:

    answer, urls = run_rag(q["question"])

    gt = q["source_url"]

    rank = urls.index(gt)+1 if gt in urls else None
    mrr = 0 if rank is None else 1/rank
    recall5 = 1 if gt in urls[:5] else 0

    f1s = f1(answer, q["answer"])

    rows.append([q["id"], mrr, recall5, f1s])

df = pd.DataFrame(rows, columns=["ID","MRR","Recall@5","F1"])
df.to_csv("results/metrics.csv", index=False)

print("Evaluation complete")
