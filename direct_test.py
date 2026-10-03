from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

question = "Represent this sentence for searching relevant passages: What is the main objective of this paper?"

abstract_text = (
    "The dominant sequence transduction models are based on complex recurrent or "
    "convolutional neural networks that include an encoder and a decoder. We propose "
    "a new simple network architecture, the Transformer, based solely on attention "
    "mechanisms, dispensing with recurrence and convolutions entirely."
)

reference_text = (
    "Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. "
    "Building a large annotated corpus of english: The penn treebank. "
    "Computational linguistics, 19(2):313-330, 1993."
)

embeddings = model.encode([question, abstract_text, reference_text])

score_abstract = util.cos_sim(embeddings[0], embeddings[1])
score_reference = util.cos_sim(embeddings[0], embeddings[2])

print("Question vs Abstract:", float(score_abstract))
print("Question vs Reference:", float(score_reference))