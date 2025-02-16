import pickle
import spacy
from sentence_transformers import SentenceTransformer

# Load NLP and BERT model
nlp = spacy.load("en_core_web_sm")
print("Model loaded successfully!")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Save models
with open("nlp_model.pkl", "wb") as f:
    pickle.dump(nlp, f)

with open("bert_model.pkl", "wb") as f:
    pickle.dump(bert_model, f)

print("Models saved successfully!")
