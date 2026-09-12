import os
import re
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd

from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# NLTK SETUP
# ==========================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# ==========================================================
# TESSERACT CONFIGURATION
# ==========================================================

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

tesseract_installed = False

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_installed = True
        break


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ==========================================================
# PROJECT PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

JOB_FILE = DATA_DIR / "job_roles.csv"


# ==========================================================
# LOAD JOB DATASET
# ==========================================================

@st.cache_data
def load_job_roles():

    if not JOB_FILE.exists():
        raise FileNotFoundError(
            f"Job dataset not found at:\n{JOB_FILE}"
        )

    try:

        df = pd.read_csv(
            JOB_FILE,
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        df = pd.read_csv(
            JOB_FILE,
            encoding="latin1"
        )

    required_columns = [
        "job_role",
        "required_skills",
        "job_description"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in job_roles.csv:\n"
            + ", ".join(missing_columns)
        )

    df = df.dropna(
        subset=[
            "job_role",
            "required_skills",
            "job_description"
        ]
    )

    return df


# ==========================================================
# TEXT EXTRACTION
# ==========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(uploaded_file):

    document = Document(uploaded_file)

    text = ""

    for paragraph in document.paragraphs:

        paragraph_text = paragraph.text.strip()

        if paragraph_text:
            text += paragraph_text + "\n"

    return text


def extract_image_text(uploaded_file):

    if not tesseract_installed:

        return None

    image = Image.open(
        uploaded_file
    )

    return pytesseract.image_to_string(
        image
    )


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        " ",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    stop_words = set(
        stopwords.words("english")
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ==========================================================
# RESUME VALIDATION
# ==========================================================

def is_valid_resume(text):

    if not text:

        return False

    if len(text.strip()) < 100:

        return False

    lower_text = text.lower()

    resume_keywords = [
        "resume",
        "curriculum vitae",
        "objective",
        "summary",
        "education",
        "skills",
        "technical skills",
        "experience",
        "work experience",
        "professional experience",
        "internship",
        "projects",
        "certifications",
        "achievements",
        "linkedin",
        "github"
    ]

    keyword_count = sum(
        1
        for keyword in resume_keywords
        if keyword in lower_text
    )

    email_pattern = (
        r"[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    phone_pattern = (
        r"(\+91[\s-]?)?[6-9]\d{9}"
    )

    has_email = bool(
        re.search(
            email_pattern,
            text
        )
    )

    has_phone = bool(
        re.search(
            phone_pattern,
            text
        )
    )

    return (
        keyword_count >= 3
        and (
            has_email
            or has_phone
        )
    )


# ==========================================================
# SKILL DATABASE
# ==========================================================

SKILL_DATABASE = [
    "Python",
    "Java",
    "C++",
    "C#",
    "JavaScript",
    "TypeScript",
    "HTML",
    "CSS",
    "React",
    "Angular",
    "Node.js",
    "Express.js",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Scikit-learn",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "NLP",
    "Natural Language Processing",
    "TensorFlow",
    "PyTorch",
    "OpenCV",
    "Git",
    "GitHub",
    "Docker",
    "AWS",
    "Azure",
    "Power BI",
    "Tableau",
    "Excel",
    "Firebase",
    "Android",
    "Kotlin",
    "Flutter",
    "Django",
    "Flask",
    "REST API",
    "Statistics"
]


# ==========================================================
# SKILL DETECTION
# ==========================================================

def detect_skills(text):

    detected_skills = []

    for skill in SKILL_DATABASE:

        pattern = re.escape(
            skill
        )

        if re.search(
            r"(?<![a-zA-Z0-9])"
            + pattern +
            r"(?![a-zA-Z0-9])",
            text,
            re.IGNORECASE
        ):

            detected_skills.append(
                skill
            )

    return list(
        dict.fromkeys(
            detected_skills
        )
    )


# ==========================================================
# JOB MATCHING
# ==========================================================

def calculate_job_matches(
    resume_text,
    resume_skills,
    job_df
):

    cleaned_resume = clean_text(
        resume_text
    )

    resume_skill_set = {
        skill.lower()
        for skill in resume_skills
    }

    results = []

    for _, row in job_df.iterrows():

        role = str(
            row["job_role"]
        ).strip()

        required_skill_text = str(
            row["required_skills"]
        )

        job_description = str(
            row["job_description"]
        )

        required_skills = [
            skill.strip()
            for skill in
            required_skill_text.split(",")
            if skill.strip()
        ]

        matched_skills = []

        missing_skills = []

        for skill in required_skills:

            if skill.lower() in resume_skill_set:

                matched_skills.append(
                    skill
                )

            else:

                missing_skills.append(
                    skill
                )

        if required_skills:

            skill_coverage = (
                len(matched_skills)
                / len(required_skills)
            ) * 100

        else:

            skill_coverage = 0

        # -----------------------------
        # TF-IDF
        # -----------------------------

        cleaned_job = clean_text(
            job_description
        )

        try:

            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2)
            )

            vectors = vectorizer.fit_transform(
                [
                    cleaned_resume,
                    cleaned_job
                ]
            )

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )[0][0]

        except ValueError:

            similarity = 0

        text_similarity = (
            similarity * 100
        )

        # -----------------------------
        # Final score
        # -----------------------------

        final_score = (
            skill_coverage * 0.75
            + text_similarity * 0.25
        )

        final_score = round(
            min(
                final_score,
                100
            )
        )

        if final_score >= 75:

            label = "Strong Match"

        elif final_score >= 55:

            label = "Good Match"

        elif final_score >= 40:

            label = "Moderate Match"

        else:

            label = "Low Match"

        results.append({

            "role": role,

            "match": final_score,

            "label": label,

            "skill_coverage":
                round(skill_coverage),

            "text_similarity":
                round(text_similarity),

            "matched_skills":
                matched_skills,

            "missing_skills":
                missing_skills
        })

    results.sort(
        key=lambda item:
            item["match"],
        reverse=True
    )

    return results


# ==========================================================
# SECTION DETECTION
# ==========================================================

def detect_section(
    text,
    keywords
):

    lower_text = text.lower()

    return any(
        keyword in lower_text
        for keyword in keywords
    )


# ==========================================================
# RESUME SCORE
# ==========================================================

def calculate_resume_score(
    skills,
    has_education,
    has_experience,
    has_projects,
    has_contact,
    has_summary,
    has_certifications
):

    score = 0

    if len(skills) >= 8:

        score += 25

    elif len(skills) >= 5:

        score += 20

    elif len(skills) >= 3:

        score += 15

    elif len(skills) >= 1:

        score += 10

    if has_education:
        score += 15

    if has_experience:
        score += 20

    if has_projects:
        score += 15

    if has_contact:
        score += 10

    if has_summary:
        score += 5

    if has_certifications:
        score += 10

    return min(
        score,
        100
    )


# ==========================================================
# PDF REPORT GENERATION
# ==========================================================

def generate_pdf_report(
    file_name,
    resume_score,
    skills,
    best_job,
    job_matches,
    has_education,
    has_experience,
    has_projects,
    has_summary,
    has_certifications,
    has_contact,
    strengths,
    improvements,
    suggestions
):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14
    )

    story = []

    story.append(
        Paragraph("AI Resume Analysis Report", title_style)
    )

    story.append(
        Paragraph(
            f"<b>Resume:</b> {file_name}",
            body_style
        )
    )

    story.append(Spacer(1, 10))

    # Overview
    story.append(
        Paragraph("Resume Overview", heading_style)
    )

    overview_data = [
        ["Metric", "Result"],
        ["Resume Score", f"{resume_score}/100"],
        ["Skills Detected", str(len(skills))],
        ["Best Job Match", f"{best_job['role']} ({best_job['match']}%)"]
    ]

    overview_table = Table(
        overview_data,
        colWidths=[2.3 * inch, 3.8 * inch]
    )

    overview_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(overview_table)

    # Skills
    story.append(
        Paragraph("Detected Skills", heading_style)
    )

    story.append(
        Paragraph(
            ", ".join(skills) if skills else "No recognized technical skills detected.",
            body_style
        )
    )

    # Best Match
    story.append(
        Paragraph("Best Job Match", heading_style)
    )

    story.append(
        Paragraph(
            f"<b>{best_job['role']}</b> — "
            f"{best_job['match']}% ({best_job['label']})",
            body_style
        )
    )

    story.append(
        Paragraph(
            "<b>Matched Skills:</b> "
            + (
                ", ".join(best_job["matched_skills"])
                if best_job["matched_skills"]
                else "None"
            ),
            body_style
        )
    )

    story.append(
        Paragraph(
            "<b>Missing Skills:</b> "
            + (
                ", ".join(best_job["missing_skills"])
                if best_job["missing_skills"]
                else "None"
            ),
            body_style
        )
    )

    # All Job Matches
    story.append(
        Paragraph("Job Role Matching", heading_style)
    )

    job_data = [
        ["Job Role", "Match", "Skill Coverage", "Text Similarity"]
    ]

    for job in job_matches:
        job_data.append([
            job["role"],
            f"{job['match']}%",
            f"{job['skill_coverage']}%",
            f"{job['text_similarity']}%"
        ])

    job_table = Table(
        job_data,
        colWidths=[2.2 * inch, 1.0 * inch, 1.2 * inch, 1.2 * inch]
    )

    job_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    story.append(job_table)

    # Resume Sections
    story.append(
        Paragraph("Resume Sections", heading_style)
    )

    section_data = [
        ["Section", "Status"],
        ["Education", "Present" if has_education else "Missing"],
        ["Experience", "Present" if has_experience else "Missing"],
        ["Projects", "Present" if has_projects else "Missing"],
        ["Summary", "Present" if has_summary else "Missing"],
        ["Certifications", "Present" if has_certifications else "Missing"],
        ["Contact", "Present" if has_contact else "Missing"],
    ]

    section_table = Table(
        section_data,
        colWidths=[2.8 * inch, 2.8 * inch]
    )

    section_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )

    story.append(section_table)

    # Strengths
    story.append(
        Paragraph("Strengths", heading_style)
    )

    for item in strengths:
        story.append(
            Paragraph(f"• {item}", body_style)
        )

    # Areas to Improve
    story.append(
        Paragraph("Areas to Improve", heading_style)
    )

    for item in improvements:
        story.append(
            Paragraph(f"• {item}", body_style)
        )

    # Suggestions
    story.append(
        Paragraph("Improvement Suggestions", heading_style)
    )

    for item in suggestions:
        story.append(
            Paragraph(f"• {item}", body_style)
        )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "📄 AI Resume Analyzer"
)

st.write(
    "Upload a professional resume and get "
    "intelligent career insights."
)

st.divider()


# ==========================================================
# LOAD CSV
# ==========================================================

try:

    job_df = load_job_roles()

except Exception as error:

    st.error(
        "❌ Unable to load job_roles.csv"
    )

    st.code(
        str(error)
    )

    st.info(
        "Expected file location:"
    )

    st.code(
        str(JOB_FILE)
    )

    st.stop()


# ==========================================================
# UPLOAD
# ==========================================================

st.subheader(
    "📤 Upload Your Resume"
)

st.info(
    "Supported formats: PDF, DOCX, JPG, JPEG and PNG."
)

uploaded_file = st.file_uploader(
    "Choose your resume",
    type=[
        "pdf",
        "docx",
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ==========================================================
# PROCESS RESUME
# ==========================================================

if uploaded_file is not None:

    try:

        file_type = uploaded_file.type

        # ----------------------------------------------
        # PDF
        # ----------------------------------------------

        if file_type == "application/pdf":

            resume_text = extract_pdf_text(
                uploaded_file
            )

        # ----------------------------------------------
        # DOCX
        # ----------------------------------------------

        elif file_type == (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        ):

            resume_text = extract_docx_text(
                uploaded_file
            )

        # ----------------------------------------------
        # IMAGE
        # ----------------------------------------------

        elif file_type in [
            "image/png",
            "image/jpeg",
            "image/jpg"
        ]:

            if not tesseract_installed:

                st.error(
                    "❌ Image resume analysis requires "
                    "Tesseract OCR."
                )

                st.info(
                    "Please install Tesseract OCR on Windows "
                    "and restart the application."
                )

                st.stop()

            resume_text = extract_image_text(
                uploaded_file
            )

        else:

            resume_text = ""

        # ----------------------------------------------
        # VALIDATE
        # ----------------------------------------------

        if not is_valid_resume(
            resume_text
        ):

            st.warning(
                "⚠️ Please upload a professional resume."
            )

            st.write(
                "The uploaded file does not appear "
                "to be a valid resume."
            )

        else:

            st.success(
                "✅ Professional resume detected."
            )

            st.divider()

            st.subheader(
                "📄 Resume"
            )

            st.write(
                f"**File:** {uploaded_file.name}"
            )

            st.divider()

            # ----------------------------------------------
            # ANALYZE BUTTON
            # ----------------------------------------------

            if st.button(
                "🔍 Analyze Resume",
                use_container_width=True
            ):

                # ------------------------------------------
                # SKILLS
                # ------------------------------------------

                skills = detect_skills(
                    resume_text
                )

                # ------------------------------------------
                # SECTIONS
                # ------------------------------------------

                has_education = detect_section(
                    resume_text,
                    [
                        "education",
                        "educational background",
                        "academic qualification"
                    ]
                )

                has_experience = detect_section(
                    resume_text,
                    [
                        "experience",
                        "work experience",
                        "professional experience",
                        "internship"
                    ]
                )

                has_projects = detect_section(
                    resume_text,
                    [
                        "projects",
                        "personal projects",
                        "academic projects"
                    ]
                )

                has_summary = detect_section(
                    resume_text,
                    [
                        "summary",
                        "professional summary",
                        "career objective",
                        "objective"
                    ]
                )

                has_certifications = detect_section(
                    resume_text,
                    [
                        "certifications",
                        "certification",
                        "courses"
                    ]
                )

                # ------------------------------------------
                # CONTACT
                # ------------------------------------------

                email_pattern = (
                    r"[A-Za-z0-9._%+-]+"
                    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
                )

                phone_pattern = (
                    r"(\+91[\s-]?)?[6-9]\d{9}"
                )

                has_contact = (
                    bool(
                        re.search(
                            email_pattern,
                            resume_text
                        )
                    )
                    or
                    bool(
                        re.search(
                            phone_pattern,
                            resume_text
                        )
                    )
                )

                # ------------------------------------------
                # SCORE
                # ------------------------------------------

                resume_score = calculate_resume_score(
                    skills,
                    has_education,
                    has_experience,
                    has_projects,
                    has_contact,
                    has_summary,
                    has_certifications
                )

                # ------------------------------------------
                # JOB MATCH
                # ------------------------------------------

                job_matches = calculate_job_matches(
                    resume_text,
                    skills,
                    job_df
                )

                # ------------------------------------------
                # NO MATCHES
                # ------------------------------------------

                if not job_matches:

                    st.warning(
                        "No job roles are available "
                        "for matching."
                    )

                    st.stop()

                best_job = job_matches[0]

                # ==================================================
                # OVERVIEW
                # ==================================================

                st.divider()

                st.subheader(
                    "📊 Resume Overview"
                )

                col1, col2, col3 = st.columns(
                    3
                )

                with col1:

                    st.metric(
                        "Resume Score",
                        f"{resume_score}/100"
                    )

                with col2:

                    st.metric(
                        "Skills Detected",
                        len(skills)
                    )

                with col3:

                    st.metric(
                        "Best Job Match",
                        f"{best_job['match']}%"
                    )

                st.progress(
                    resume_score / 100
                )

                # ==================================================
                # SKILLS
                # ==================================================

                st.subheader(
                    "🧠 Skills Detected"
                )

                if skills:

                    st.success(
                        " • ".join(
                            skills
                        )
                    )

                else:

                    st.warning(
                        "No recognized technical skills detected."
                    )

                # ==================================================
                # BEST JOB MATCH
                # ==================================================

                st.subheader(
                    "🎯 Best Job Match"
                )

                if best_job["match"] >= 40:

                    st.write(
                        f"### {best_job['role']}"
                    )

                    st.metric(
                        "Match Percentage",
                        f"{best_job['match']}%"
                    )

                    st.info(
                        best_job["label"]
                    )

                    st.progress(
                        best_job["match"] / 100
                    )

                else:

                    st.warning(
                        "⚠️ No strong job match found."
                    )

                    st.write(
                        "The resume does not currently "
                        "show enough relevant skills "
                        "for the available roles."
                    )

                # ==================================================
                # MATCHED SKILLS
                # ==================================================

                st.subheader(
                    "✅ Matched Skills"
                )

                if best_job[
                    "matched_skills"
                ]:

                    for skill in best_job[
                        "matched_skills"
                    ]:

                        st.write(
                            f"✅ {skill}"
                        )

                else:

                    st.write(
                        "No direct skill matches detected."
                    )

                # ==================================================
                # MISSING SKILLS
                # ==================================================

                if best_job[
                    "missing_skills"
                ]:

                    st.subheader(
                        "❌ Missing Skills"
                    )

                    st.warning(
                        "Your resume is missing some "
                        "skills required for this role."
                    )

                    for skill in best_job[
                        "missing_skills"
                    ]:

                        st.write(
                            f"❌ {skill}"
                        )

                    st.subheader(
                        "📚 Recommended Skills"
                    )

                    st.info(
                        "Consider learning these skills "
                        "to improve your job match:"
                    )

                    for skill in best_job[
                        "missing_skills"
                    ]:

                        st.write(
                            f"📌 {skill}"
                        )

                else:

                    st.subheader(
                        "✅ Skill Requirements Met"
                    )

                    st.success(
                        "Great! Your resume covers all "
                        "the key skills required for "
                        "this role."
                    )

                    st.subheader(
                        "🎯 Recommendation"
                    )

                    st.info(
                        "No additional technical skills "
                        "are required for this role at "
                        "this time. Focus on strengthening "
                        "your projects and practical experience."
                    )

                # ==================================================
                # ALL JOB MATCHES
                # ==================================================

                st.subheader(
                    "💼 Job Role Matching"
                )

                for job in job_matches:

                    st.write(
                        f"**{job['role']} — "
                        f"{job['match']}%**"
                    )

                    st.caption(
                        f"{job['label']} | "
                        f"Skill Coverage: "
                        f"{job['skill_coverage']}% | "
                        f"Text Similarity: "
                        f"{job['text_similarity']}%"
                    )

                    st.progress(
                        job["match"] / 100
                    )

                    if job[
                        "matched_skills"
                    ]:

                        st.write(
                            "**Matched:** "
                            + ", ".join(
                                job[
                                    "matched_skills"
                                ]
                            )
                        )

                    if job[
                        "missing_skills"
                    ]:

                        st.write(
                            "**Missing:** "
                            + ", ".join(
                                job[
                                    "missing_skills"
                                ]
                            )
                        )

                    st.divider()

                # ==================================================
                # RESUME SECTIONS
                # ==================================================

                st.subheader(
                    "📋 Resume Sections"
                )

                c1, c2, c3 = st.columns(
                    3
                )

                with c1:

                    if has_education:
                        st.success(
                            "✅ Education"
                        )
                    else:
                        st.error(
                            "❌ Education"
                        )

                with c2:

                    if has_experience:
                        st.success(
                            "✅ Experience"
                        )
                    else:
                        st.error(
                            "❌ Experience"
                        )

                with c3:

                    if has_projects:
                        st.success(
                            "✅ Projects"
                        )
                    else:
                        st.error(
                            "❌ Projects"
                        )

                c4, c5, c6 = st.columns(
                    3
                )

                with c4:

                    if has_summary:
                        st.success(
                            "✅ Summary"
                        )
                    else:
                        st.error(
                            "❌ Summary"
                        )

                with c5:

                    if has_certifications:
                        st.success(
                            "✅ Certifications"
                        )
                    else:
                        st.error(
                            "❌ Certifications"
                        )

                with c6:

                    if has_contact:
                        st.success(
                            "✅ Contact"
                        )
                    else:
                        st.error(
                            "❌ Contact"
                        )

                # ==================================================
                # STRENGTHS
                # ==================================================

                st.subheader(
                    "✅ Strengths"
                )

                strengths = []

                if len(skills) >= 5:
                    strengths.append(
                        "Good technical skill coverage."
                    )

                if has_education:
                    strengths.append(
                        "Education section is included."
                    )

                if has_experience:
                    strengths.append(
                        "Experience section is included."
                    )

                if has_projects:
                    strengths.append(
                        "Projects section is included."
                    )

                if has_certifications:
                    strengths.append(
                        "Certifications are included."
                    )

                if has_summary:
                    strengths.append(
                        "Professional summary is included."
                    )

                if has_contact:
                    strengths.append(
                        "Contact information is available."
                    )

                if not strengths:
                    strengths.append(
                        "Basic resume structure detected."
                    )

                for item in strengths:

                    st.write(
                        f"✅ {item}"
                    )

                # ==================================================
                # AREAS TO IMPROVE
                # ==================================================

                st.subheader(
                    "⚠️ Areas to Improve"
                )

                improvements = []

                if len(skills) < 5:
                    improvements.append(
                        "Add more relevant technical skills."
                    )

                if not has_experience:
                    improvements.append(
                        "Add internship or work experience."
                    )

                if not has_projects:
                    improvements.append(
                        "Add relevant projects."
                    )

                if not has_education:
                    improvements.append(
                        "Add complete education details."
                    )

                if not has_summary:
                    improvements.append(
                        "Add a professional summary."
                    )

                if not has_certifications:
                    improvements.append(
                        "Add relevant certifications or courses."
                    )

                if not has_contact:
                    improvements.append(
                        "Add complete contact information."
                    )

                if not improvements:
                    improvements.append(
                        "Your resume has a strong basic structure."
                    )

                for item in improvements:

                    st.write(
                        f"⚠️ {item}"
                    )

                # ==================================================
                # IMPROVEMENT SUGGESTIONS
                # ==================================================

                st.subheader(
                    "💡 Improvement Suggestions"
                )

                suggestions = [
                    "Use clear and professional section headings.",
                    "Keep skills relevant to your target role.",
                    "Describe projects using technologies and measurable results.",
                    "Use action-oriented language for experience.",
                    "Add GitHub and LinkedIn links when appropriate.",
                    "Keep the resume concise and easy to scan.",
                    "Avoid unnecessary personal information."
                ]

                for suggestion in suggestions:

                    st.write(
                        f"• {suggestion}"
                    )

                # ==================================================
                # DOWNLOAD ANALYSIS REPORT
                # ==================================================

                try:

                    report_pdf = generate_pdf_report(
                        file_name=uploaded_file.name,
                        resume_score=resume_score,
                        skills=skills,
                        best_job=best_job,
                        job_matches=job_matches,
                        has_education=has_education,
                        has_experience=has_experience,
                        has_projects=has_projects,
                        has_summary=has_summary,
                        has_certifications=has_certifications,
                        has_contact=has_contact,
                        strengths=strengths,
                        improvements=improvements,
                        suggestions=suggestions
                    )

                    st.divider()

                    st.subheader(
                        "📥 Download Analysis Report"
                    )

                    st.success(
                        "Your analysis report is ready."
                    )

                    report_name = (
                        Path(uploaded_file.name).stem
                        + "_AI_Resume_Analysis_Report.pdf"
                    )

                    st.download_button(
                        label="📥 Download PDF Report",
                        data=report_pdf,
                        file_name=report_name,
                        mime="application/pdf",
                        use_container_width=True
                    )

                except Exception as report_error:

                    st.error(
                        "❌ Could not generate the PDF report."
                    )

                    st.code(
                        str(report_error)
                    )

    except Exception as error:

        st.error(
            "❌ Unable to process this file."
        )

        st.code(
            str(error)
        )