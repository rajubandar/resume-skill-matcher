from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
resume = open("sample_resume.txt", "r").read()
jd = open("sample_jd.txt", "r").read()

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform([resume, jd])

# Similarity score
score = cosine_similarity(vectors[0], vectors[1])[0][0]
print(f"\nResume–JD Match Score: {score*100:.2f}%")

# Skill extraction (basic keyword approach)
skills = [
    "python", "machine learning", "deep learning", "nlp",
    "sql", "data analysis", "cloud", "git"
]

resume_lower = resume.lower()
jd_lower = jd.lower()

resume_skills = [s for s in skills if s in resume_lower]
jd_skills = [s for s in skills if s in jd_lower]

missing_skills = set(jd_skills) - set(resume_skills)

print("\nSkills Found in Resume:", resume_skills)
print("Skills Required in JD:", jd_skills)
print("Missing Skills:", list(missing_skills))
