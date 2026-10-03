import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("pdf_chunks")


def get_page1_chunks(limit=2):
    all_data = collection.get(include=["documents", "metadatas"])
    page1 = [
        {"text": doc, "page": meta["page"]}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        if meta["page"] == 1
    ]
    return page1[:limit]


def get_relevant_chunks(question, n_results=6):
    query_text = "Represent this sentence for searching relevant passages: " + question
    question_embedding = embed_model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
        })

    page1_chunks = get_page1_chunks()
    existing_texts = {c["text"] for c in chunks}
    for c in page1_chunks:
        if c["text"] not in existing_texts:
            chunks.append(c)

    return chunks


def build_prompt(question, chunks):
    context = ""
    for chunk in chunks:
        context += f"[Page {chunk['page']}]\n{chunk['text']}\n\n"

    prompt = f"""Answer the question using ONLY the context below. \
If the answer is not in the context, say "I couldn't find this in the document."
Mention the page number(s) your answer comes from.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask(question):
    chunks = get_relevant_chunks(question)
    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type quit): ")
        if question.strip().lower() == "quit":
            break
        answer = ask(question)
        print("\n" + answer)