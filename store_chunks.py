import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 500
MIN_CHUNK_LENGTH = 200


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": page_number, "text": text})
    return pages


def split_text(text, chunk_size, overlap):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period != -1 and last_period > start + chunk_size // 2:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks


pages = load_pdf("sample.pdf")

all_chunks = []
for page in pages:
    for piece in split_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP):
        if len(piece) >= MIN_CHUNK_LENGTH:
            all_chunks.append({"page": page["page"], "text": piece})

print("Total chunks:", len(all_chunks))

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection("pdf_chunks")
except Exception:
    pass

collection = client.create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"},
)

ids = [f"chunk_{i}" for i in range(len(all_chunks))]
metadatas = [{"page": chunk["page"]} for chunk in all_chunks]

for i in range(0, len(texts), BATCH_SIZE):
    collection.add(
        ids=ids[i : i + BATCH_SIZE],
        documents=texts[i : i + BATCH_SIZE],
        embeddings=embeddings[i : i + BATCH_SIZE].tolist(),
        metadatas=metadatas[i : i + BATCH_SIZE],
    )

print("Stored in ChromaDB:", collection.count(), "chunks")