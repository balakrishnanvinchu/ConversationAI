from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to("cpu")


def generate_answer(query, contexts):

    cleaned = [c[:800] for c in contexts[:3]]

    context_text = "\n".join(cleaned)

    prompt = f"""
Answer strictly from context.

Context:
{context_text}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
