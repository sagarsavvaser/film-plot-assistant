import os, glob, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

def load_docs(folder="films"):
    docs = []
    for path in glob.glob(os.path.join(folder, "*.txt*")):
        title = os.path.basename(path)
        title = re.sub(r"(\.txt)+$", "", title, flags=re.IGNORECASE)

        with open(path, encoding="utf-8", errors="ignore") as f:
            text = re.sub(r"\[\d+\]", "", f.read())

        if text.strip():
            docs.append((title, text))

    if not docs:
        raise FileNotFoundError(
            "No TXT files found. Check that the films folder exists."
        )

    return docs

def chunk_docs(docs, size=200, overlap=40):
    chunks = []
    step = size - overlap

    for title, text in docs:
        words = text.split()

        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + size])

            if len(piece.split()) > 30:
                chunks.append({"text": piece, "title": title})

    if not chunks:
        raise ValueError("No text chunks created. Check your movie TXT files.")

    return chunks

def build_index(chunks):
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    mat = vec.fit_transform(
        [f"{c['title']} {c['text']}" for c in chunks]
    )
    return vec, mat

def retrieve(question, index, chunks, k=4):
    vec, mat = index
    sims = cosine_similarity(vec.transform([question]), mat)[0]
    top = sims.argsort()[::-1][:k]
    return [chunks[i] for i in top if sims[i] > 0]

PROMPT = """You are a movie assistant. Answer ONLY using the context below.
Mention the film title(s) you used, like (Skyfall). If the answer is not in the context, reply exactly:
"Not found in the films."

Context:
{context}

Question: {question}"""

def answer(question, hits, api_key, model):
    if not hits:
        return "Not found in the films."

    client = genai.Client(api_key=api_key)
    context = "\n\n".join(
        f"[{h['title']}]\n{h['text']}" for h in hits
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=PROMPT.format(
                    context=context,
                    question=question
                )
            )
            return response.text or "Not found in the films."

        except Exception as e:
            if attempt == 2:
                return f"Error: {e}"
            time.sleep(8 * (attempt + 1))
