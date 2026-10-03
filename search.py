import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pdf_chunks")


def keyword_score(question, text):
    question_words = set(question.lower().split())
    text_lower = text.lower()
    matches = sum(1 for w in question_words if len(w) > 3 and w in text_lower)
    return matches


while True:
    question = input("\nAsk a question (or type quit): ")
    if question.strip().lower() == "quit":
        break

    query_text = "Represent this sentence for searching relevant passages: " + question
    question_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=20,
    )

    combined = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        page = results["metadatas"][0][i]["page"]
        distance = results["distances"][0][i]
        kw_score = keyword_score(question, text)
        combined.append({
            "text": text,
            "page": page,
            "distance": distance,
            "kw_score": kw_score,
        })

    combined.sort(key=lambda c: (-c["kw_score"], c["distance"]))

    for i, c in enumerate(combined[:5]):
        print(f"\n--- Result {i + 1} (page {c['page']}, distance {c['distance']:.3f}, keyword matches {c['kw_score']}) ---")
        print(c["text"])