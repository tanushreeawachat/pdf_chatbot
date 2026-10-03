import os
import tempfile
import streamlit as st
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 200


@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return embed_model, groq_client


embed_model, groq_client = load_models()


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


def process_pdf(path):
    pages = load_pdf(path)
    all_chunks = []
    for page in pages:
        for piece in split_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            if len(piece) >= MIN_CHUNK_LENGTH:
                all_chunks.append({"page": page["page"], "text": piece})

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False)

    client = chromadb.Client()
    try:
        client.delete_collection("pdf_chunks")
    except Exception:
        pass
    collection = client.create_collection(
        name="pdf_chunks", metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"page": c["page"]} for c in all_chunks]
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    return collection


def get_page1_chunks(collection, limit=2):
    all_data = collection.get(include=["documents", "metadatas"])
    page1 = [
        {"text": doc, "page": meta["page"]}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        if meta["page"] == 1
    ]
    return page1[:limit]


def get_relevant_chunks(collection, question, n_results=6):
    query_text = "Represent this sentence for searching relevant passages: " + question
    question_embedding = embed_model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=n_results)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
        })

    existing_texts = {c["text"] for c in chunks}
    for c in get_page1_chunks(collection):
        if c["text"] not in existing_texts:
            chunks.append(c)

    return chunks


def build_prompt(question, chunks):
    context = ""
    for chunk in chunks:
        context += f"[Page {chunk['page']}]\n{chunk['text']}\n\n"

    return f"""Answer the question using ONLY the context below. \
If the answer is not in the context, say "I couldn't find this in the document."
Mention the page number(s) your answer comes from.

Context:
{context}

Question: {question}

Answer:"""


def ask(collection, question):
    chunks = get_relevant_chunks(collection, question)
    prompt = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    with st.spinner(f"Processing {uploaded_file.name}..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.session_state.collection = process_pdf(tmp_path)
        os.unlink(tmp_path)
        st.session_state.filename = uploaded_file.name
        st.session_state.messages = []
    st.success(f"Ready! Ask questions about {uploaded_file.name}")

if st.session_state.collection is not None:
    st.caption(f"Chatting with: {st.session_state.filename}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the PDF...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(st.session_state.collection, question)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF above to get started.")