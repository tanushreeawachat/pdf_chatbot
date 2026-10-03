import datetime
import chromadb
from sentence_transformers import SentenceTransformer

print("Run time:", datetime.datetime.now())

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Model loaded:", model)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pdf_chunks")
print("Chunks in DB:", collection.count())

query_text = "Represent this sentence for searching relevant passages: What is the main objective of this paper?"
question_embedding = model.encode([query_text]).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=5,
)

for i in range(len(results["documents"][0])):
    text = results["documents"][0][i]
    page = results["metadatas"][0][i]["page"]
    distance = results["distances"][0][i]
    print(f"\n--- Result {i + 1} (page {page}, distance {distance:.4f}) ---")
    print(text[:200])