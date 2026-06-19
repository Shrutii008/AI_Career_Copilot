import streamlit as st
import pdfplumber
import base64
import time

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🤖 AI Resume Analyzer")
    st.markdown("### Your AI-Powered ATS Assistant")

    st.markdown("---")
    st.subheader("✨ Features")

    st.write("📄 PDF Resume Parsing")
    st.write("🧠 Technical Skill Detection")
    st.write("📊 ATS Compatibility Score")
    st.write("⚠️ Missing Skill Suggestions")
    st.write("❓ AI Interview Questions")
    st.write("📝 Resume Summary")

    st.markdown("---")
    st.info(
        "Upload your resume and receive instant AI-powered insights."
    )

# ---------------- MAIN HEADER ----------------

st.title("🤖 AI Resume Analyzer")

st.markdown(
    """
### Analyze your resume like an ATS system 🚀

Check your skills, improve your resume,
and prepare better for interviews.
"""
)

st.markdown("---")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📂 Upload Your Resume (PDF)",
    type=["pdf"]
)

# ---------------- SKILL DATABASE ----------------

skills_db = [
    "python",
    "java",
    "sql",
    "machine learning",
    "deep learning",
    "git",
    "tensorflow",
    "pandas",
    "numpy"
]

recommended = [
    "python",
    "sql",
    "git",
    "docker",
    "aws"
]

if uploaded_file is not None:

    with st.spinner("🔍 AI is analyzing your resume..."):
        time.sleep(2)

        text = ""

        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    text_lower = text.lower()

    # ---------------- SKILL DETECTION ----------------

    found_skills = []

    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)

    # ---------------- ATS SCORE ----------------

    score = len(found_skills) * 10

    if score > 100:
        score = 100
        missing = []

    for skill in recommended:
        if skill not in found_skills:
            missing.append(skill)

    # ---------------- INTERVIEW QUESTIONS ----------------

    questions = []

    if "python" in found_skills:
        questions.append("What is OOP in Python?")

    if "sql" in found_skills:
        questions.append("What is a JOIN in SQL?")

    if "machine learning" in found_skills:
        questions.append("What is overfitting?")

    # ---------------- SUMMARY ----------------

    if found_skills:
        summary = "Candidate has skills in: " + ", ".join(found_skills)
    else:
        summary = "No strong skills detected"

    # ---------------- PAGE LAYOUT ----------------

    col1, col2 = st.columns([2, 1])

    # ---------------- LEFT SIDE ----------------

    with col1:

        st.subheader("📄 Resume Content")
        st.write(text)

        st.subheader("👤 Candidate Details")

        name = text.split("\n")[0] if text else "Not Found"

        email = "Found" if "@" in text else "Not Found"

        phone = (
            "Found"
            if any(char.isdigit() for char in text)
            else "Not Found"
        )

        st.success(f"👤 Name: {name}")
        st.success(f"📧 Email: {email}")
        st.success(f"📱 Phone: {phone}")


    # ---------------- RIGHT DASHBOARD ----------------

    with col2:

        st.subheader("📊 ATS Dashboard")

        st.markdown("### 🧠 Skills Detected")

        if found_skills:
            for skill in found_skills:
                st.success(skill.title())
        else:
            st.error("No skills detected")

        st.markdown("### 🎯 ATS Score")

        st.progress(score / 100)

        st.metric(
            label="Compatibility Score",
            value=f"{score}%"
        )

        st.markdown("### ⚠️ Missing Skills")

        if missing:
            for skill in missing:
                st.warning(skill.title())
        else:
            st.success("Excellent! No missing skills.")

        st.markdown("### ❓ Interview Questions")

        if questions:
            for q in questions:
                st.info(q)
        else:
            st.write("No questions generated")
            st.markdown("### 📝 Resume Summary")
        st.success(summary)

    # ---------------- DOWNLOAD REPORT ----------------

    st.markdown("---")

    st.subheader("📥 Download Resume Analysis Report")

    job_role = st.selectbox(
        "🎯 Select Your Target Job Role",
        [
            "Data Analyst",
            "ML Engineer",
            "Software Developer"
        ]
    )

    report_text = f"""
AI Resume Analysis Report

Candidate Details
-----------------
Name: {name}
Email: {email}
Phone: {phone}

Job Role:
{job_role}

Skills Detected:
{", ".join(found_skills) if found_skills else "No skills detected"}

ATS Score:
{score}%

Missing Skills:
{", ".join(missing) if missing else "None"}

Interview Questions:
{chr(10).join(questions) if questions else "No questions generated"}

Summary:
{summary}

Generated by AI Resume Analyzer
"""

    # Convert report into downloadable format

    b64 = base64.b64encode(
        report_text.encode()
    ).decode()

    download_link = (
        f'<a href="data:file/txt;base64,{b64}" '
        f'download="AI_Resume_Report.txt">'
        "📄 Click here to Download Report"
        "</a>"
    )

    st.markdown(
        download_link,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <center>
    🤖 <b>AI Resume Analyzer</b><br>
    Built with Streamlit & Python by Shruti Verma 🚀
    </center>
    """,
    unsafe_allow_html=True
)
