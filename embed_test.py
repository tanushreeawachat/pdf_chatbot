from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I cancel my subscription?",
    "Steps to terminate the agreement",
    "The weather is sunny today",
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)
print("First 5 numbers of sentence 1:", embeddings[0][:5])