from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(query, contexts):

    prompt = "Context:\n"

    for c in contexts:
        prompt += c + "\n"

    prompt += f"\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048)

    outputs = model.generate(**inputs, max_new_tokens=150)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
