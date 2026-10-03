import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pdf_chunks")

results = collection.get(include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    if meta["page"] == 1:
        print("-----")
        print(doc)
        print()