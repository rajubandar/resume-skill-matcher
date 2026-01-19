import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_resume_jd(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]

    skills = [
        "python", "machine learning", "deep learning", "nlp",
        "sql", "data analysis", "cloud", "git"
    ]

    resume_skills = [s for s in skills if s in resume_text]
    jd_skills = [s for s in skills if s in jd_text]
    missing_skills = list(set(jd_skills) - set(resume_skills))

    return score, resume_skills, jd_skills, missing_skills
