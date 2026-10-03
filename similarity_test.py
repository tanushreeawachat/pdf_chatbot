from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I cancel my subscription?",
    "Steps to terminate the agreement",
    "The weather is sunny today",
]

embeddings = model.encode(sentences)

score_1 = util.cos_sim(embeddings[0], embeddings[1])
score_2 = util.cos_sim(embeddings[0], embeddings[2])

print("Cancel vs terminate:", float(score_1))
print("Cancel vs weather:  ", float(score_2))