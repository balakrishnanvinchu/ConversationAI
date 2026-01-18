

import json, requests, re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

def extract_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.text for p in paragraphs)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()



def chunk_text(text, size=300, overlap=50):
    print("Using word_tokenize from nltk.tokenize")
    tokens = word_tokenize(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap

    return chunks


if __name__ == "__main__":

    with open("data/fixed_urls.json") as f:
        fixed = json.load(f)

    with open("data/random_urls.json") as f:
        randoms = json.load(f)

    urls = fixed + randoms

    corpus = []

    for idx, url in enumerate(urls):
        print(f"Processing {idx+1}/{len(urls)}")

        try:
            text = extract_text(url)
            if len(text.split()) < 200:
                continue

            chunks = chunk_text(text)

            for i, ch in enumerate(chunks):
                corpus.append({
                    "chunk_id": f"{idx}_{i}",
                    "url": url,
                    "text": ch
                })

        except Exception as e:
            print("Skipping:", url)
            print("Skipping:", e)

    with open("data/corpus_chunks.json", "w") as f:
        json.dump(corpus, f, indent=2)

    print("Corpus saved")
