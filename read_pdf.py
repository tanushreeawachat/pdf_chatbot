from pypdf import PdfReader

reader = PdfReader("sample.pdf")

print("Number of pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    print(f"--- Page {page_number} ---")
    print(text)