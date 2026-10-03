from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": page_number, "text": text})
    return pages


def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


pages = load_pdf("sample.pdf")

all_chunks = []
for page in pages:
    for piece in split_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP):
        all_chunks.append({"page": page["page"], "text": piece})

print("Pages with text:", len(pages))
print("Total chunks:", len(all_chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)