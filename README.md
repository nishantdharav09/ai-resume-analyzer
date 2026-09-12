# 📄 AI Resume Analyzer

AI-powered Resume Analyzer built using Python, NLP, Scikit-learn, Pandas and Streamlit.

Upload a professional resume and get useful career insights such as resume score, skills detection, job-role matching, missing skills and improvement suggestions.

## 🚀 Features

- 📤 Resume Upload
- 📊 Resume Score
- 🧠 Skills Detection
- 🎯 Best Job Match
- 💼 Job Role Matching
- ❌ Missing Skills Detection
- 📚 Recommended Skills
- ✅ Resume Strengths
- ⚠️ Areas to Improve
- 💡 Improvement Suggestions
- 📥 PDF Analysis Report

## 🛠️ Technologies Used

- Python
- Streamlit
- NLP
- NLTK
- Scikit-learn
- Pandas
- PyPDF2
- python-docx
- Pillow
- Tesseract OCR
- ReportLab

## 📁 Project Structure

```text
AI-Resume-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── run_ai_resume_analyzer.bat
│
├── data/
│   └── job_roles.csv
│
└── utils/

⚙️ How to Run

Option 1 — Run Using BAT File
Simply double-click:
run_ai_resume_analyzer.bat
The application will start automatically.

Option 2 — Run Manually
Open the terminal inside the project folder.

Step 1 — Activate Virtual Environment
venv\Scripts\activateStep 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Run the Application
python -m streamlit run app.py
The application will open in your browser.

📄 Supported Resume Formats
PDF
DOCX
JPG
JPEG
PNG

🔍 How It Works

Upload Resume
      ↓
Resume Validation
      ↓
Text Extraction
      ↓
NLP Processing
      ↓
Skills Detection
      ↓
Resume Score
      ↓
Job Role Matching
      ↓
Missing Skills
      ↓
Improvement Suggestions
      ↓
PDF Analysis Report


🎯 Job Matching

The application uses:

Skill Coverage
TF-IDF
Cosine Similarity

to calculate compatibility between the resume and available job roles.

📊 Resume Analysis

The application analyzes:

Technical Skills
Education
Experience
Projects
Certifications
Contact Information
Professional Summary

📥 PDF Report

After analyzing a resume, users can download a PDF report containing:

Resume Score
Detected Skills
Best Job Match
Matched Skills
Missing Skills
Job Role Matches
Resume Sections
Strengths
Areas to Improve
Recommendations

🔐 Privacy

Do not upload or commit private resumes containing personal information to the GitHub repository.

📌 Note

The job match percentage is an estimated score based on the implemented matching algorithm. It is not a guarantee of employment or an official ATS score.

👨‍💻 Author

Nishant Dharav
Python | Machine Learning | NLP | Streamlit

⭐ Project Goal

The goal of this project is to help users understand their resume quality, identify relevant job roles, discover skill gaps and improve their career profile.



