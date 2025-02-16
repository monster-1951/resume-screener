import streamlit as st
import pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from io import BytesIO

# Load saved models
with open("nlp_model.pkl", "rb") as f:
    nlp = pickle.load(f)

with open("bert_model.pkl", "rb") as f:
    bert_model = pickle.load(f)

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def compute_similarity_bert(resume_text, job_text):
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = bert_model.encode(job_text, convert_to_tensor=True)
    similarity = cosine_similarity([resume_embedding.cpu().numpy()], [job_embedding.cpu().numpy()])[0][0]
    return round(similarity * 100, 2)

st.title("AI Resume Screener")

uploaded_resume = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description")

if uploaded_resume and job_description:
    resume_bytes = uploaded_resume.read()

    if uploaded_resume.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_bytes)
    elif uploaded_resume.name.endswith(".docx"):
        resume_text = extract_text_from_docx(BytesIO(resume_bytes))
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
        resume_text = ""

    if resume_text:
        match_score = compute_similarity_bert(resume_text, job_description)
        st.write("### Resume Match Score:", match_score, "%")
