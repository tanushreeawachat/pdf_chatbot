from pypdf import PdfReader

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
    pieces = split_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)
    for piece in pieces:
        all_chunks.append({"page": page["page"], "text": piece})

print("Pages with text:", len(pages))
print("Total chunks:", len(all_chunks))
print()
print("--- Example chunk (chunk #3) ---")
print("From page:", all_chunks[2]["page"])
print(all_chunks[2]["text"])