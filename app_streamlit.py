import re
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #020617, #0f172a);
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 40px;
    font-size: 16px;
}

.score-container {
    background: linear-gradient(135deg, #020617, #0f172a);
    padding: 40px;
    border-radius: 28px;
    margin-top: 30px;
}

.score-pill {
    background: white;
    height: 58px;
    border-radius: 999px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    margin-bottom: 18px;
}

.score-text {
    font-size: 24px;
    font-weight: 700;
    color: #22c55e;
    display: flex;
    align-items: center;
    gap: 10px;
}

.skill-chip {
    display: inline-block;
    padding: 6px 14px;
    margin: 6px;
    border-radius: 20px;
    background: #1e293b;
    color: #93c5fd;
    font-size: 14px;
}

.missing-chip {
    background: #3f1d1d;
    color: #fca5a5;
}

.stButton > button {
    background: linear-gradient(135deg, #5f2eea, #3a86ff);
    color: white;
    font-size: 18px;
    padding: 12px 32px;
    border-radius: 30px;
    border: none;
    box-shadow: 0 10px 25px rgba(95,46,234,0.35);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 14px 35px rgba(95,46,234,0.5);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Helper Functions ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---------------- UI ----------------
st.markdown('<div class="main-title">📄 AI Resume–JD Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your resume and job description to check compatibility</div>', unsafe_allow_html=True)

resume_pdf = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
jd_pdf = st.file_uploader("📎 Upload Job Description (PDF)", type=["pdf"])

analyze = st.button("✨ Analyze Match")

# ---------------- Logic ----------------
if analyze:
    if resume_pdf and jd_pdf:
        resume_text = clean_text(extract_text_from_pdf(resume_pdf))
        jd_text = clean_text(extract_text_from_pdf(jd_pdf))

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

        # ---------------- Result UI ----------------
        st.markdown(f"""
        <div class="score-container">
            <div class="score-pill"></div>
            <div class="score-text">🎯 Match Score: {score*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("### ✅ Skills Found")
        for s in resume_skills:
            st.markdown(f'<span class="skill-chip">{s}</span>', unsafe_allow_html=True)

        st.write("### ❌ Missing Skills")
        for s in missing_skills:
            st.markdown(f'<span class="skill-chip missing-chip">{s}</span>', unsafe_allow_html=True)

    else:
        st.warning("Please upload both Resume and Job Description PDFs.")
