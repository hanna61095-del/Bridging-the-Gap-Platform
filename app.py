"""
Simple resume upload + employer job posting + keyword-overlap matching platform.

Usage:
    1) pip install -r requirements.txt
    2) python -m nltk.downloader stopwords   # one-time to fetch stopwords (or see notes below)
    3) flask run
"""

import os
import re
import uuid
from collections import Counter
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# Resume parsing libs
from pdfminer.high_level import extract_text as extract_text_pdf
import docx  # python-docx
import nltk
from nltk.corpus import stopwords

# Ensure nltk stopwords are available (if not, instruct user)
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    # If not downloaded, give a clear message when running
    STOPWORDS = set()
    print("NLTK stopwords not found. Run: python -m nltk.downloader stopwords")

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me-to-a-random-secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

db = SQLAlchemy(app)

# -----------------------
# Database models
# -----------------------
class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String, nullable=False)
    original_name = db.Column(db.String, nullable=False)
    text = db.Column(db.Text, nullable=False)  # extracted text
    candidate_name = db.Column(db.String, nullable=True)

class Employer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String, nullable=False)
    contact_email = db.Column(db.String, nullable=True)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employer_id = db.Column(db.Integer, db.ForeignKey("employer.id"), nullable=False)
    title = db.Column(db.String, nullable=False)
    description = db.Column(db.Text, nullable=False)

    employer = db.relationship("Employer", backref="jobs")

# -----------------------
# Helpers
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(path):
    doc = docx.Document(path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_pdf(path):
    try:
        text = extract_text_pdf(path)
        return text or ""
    except Exception as e:
        print("PDF extraction error:", e)
        return ""

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(path, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext in ("docx", "doc"):
        # python-docx only supports docx; .doc fallback could be improved with antiword or textract
        if ext == "doc":
            # save as binary and try reading as txt fallback
            return extract_text_from_txt(path)
        return extract_text_from_docx(path)
    elif ext == "txt":
        return extract_text_from_txt(path)
    else:
        return ""

def tokenize_and_weight(text, top_n=None):
    """
    Very simple tokenizer & weighting:
      - lowercase, remove non-letters, split
      - remove stopwords
      - return Counter of tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-+#]", " ", text)  # keep some symbols used in tech like c++, c# but simplify
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    cnt = Counter(tokens)
    if top_n:
        return dict(cnt.most_common(top_n))
    return dict(cnt)

def score_resume_against_job(resume_text, job_desc):
    """
    Simple overlap score: sum of min(count in resume, count in job) for tokens appearing in both,
    normalized by job token counts to favor resumes that match job keywords.
    Returns a numeric score (higher is better) and matched keywords.
    """
    r_tokens = tokenize_and_weight(resume_text)
    j_tokens = tokenize_and_weight(job_desc)
    if not j_tokens:
        return 0.0, {}
    overlap = 0
    matched = {}
    for tok, jcount in j_tokens.items():
        rcount = r_tokens.get(tok, 0)
        if rcount > 0:
            contribution = min(rcount, jcount)
            overlap += contribution
            matched[tok] = contribution
    # normalize by total job token counts to get a ratio (0..1+)
    total_job = sum(j_tokens.values()) or 1
    score = overlap / total_job
    return score, matched

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    jobs = Job.query.order_by(Job.id.desc()).limit(10).all()
    return render_template("index.html", jobs=jobs)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

@app.route("/upload_resume", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        candidate_name = request.form.get("candidate_name", "").strip()
        file = request.files.get("resume")
        if not file or file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("File type not allowed. Allowed: pdf, docx, doc, txt", "danger")
            return redirect(request.url)

        original_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(path)

        text = extract_text(path, original_name)
        if not text.strip():
            flash("Could not extract text from the resume. Try a different format.", "warning")

        resume = Resume(filename=unique_name, original_name=original_name, text=text, candidate_name=candidate_name)
        db.session.add(resume)
        db.session.commit()
        flash("Resume uploaded successfully.", "success")
        return redirect(url_for("match_resume", resume_id=resume.id))
    return render_template("upload_resume.html")

@app.route("/post_job", methods=["GET", "POST"])
def post_job():
    if request.method == "POST":
        company = request.form.get("company", "").strip()
        email = request.form.get("email", "").strip()
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        if not all([company, title, description]):
            flash("Company, Title and Description are required.", "danger")
            return redirect(request.url)
        employer = Employer(company=company, contact_email=email)
        db.session.add(employer)
        db.session.commit()
        job = Job(employer_id=employer.id, title=title, description=description)
        db.session.add(job)
        db.session.commit()
        flash("Job posted successfully.", "success")
        return redirect(url_for("index"))
    return render_template("post_job.html")

@app.route("/match_resume/<int:resume_id>")
def match_resume(resume_id):
    resume = Resume.query.get_or_404(resume_id)
    jobs = Job.query.all()
    scored = []
    for job in jobs:
        score, matched = score_resume_against_job(resume.text, job.description + " " + job.title)
        scored.append({
            "job": job,
            "score": score,
            "matched": matched
        })
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    # Return top 10 matches
    top = scored_sorted[:10]
    return render_template("matches.html", resume=resume, matches=top)

# convenience route to see all resumes & jobs
@app.route("/admin")
def admin_view():
    resumes = Resume.query.order_by(Resume.id.desc()).all()
    jobs = Job.query.order_by(Job.id.desc()).all()
    return render_template("admin.html", resumes=resumes, jobs=jobs)

# -----------------------
# CLI helpers
# -----------------------
@app.cli.command("init-db")
def init_db():
    """Initialize the database (run once)."""
    db.create_all()
    print("DB initialized at", app.config["SQLALCHEMY_DATABASE_URI"])

if __name__ == "__main__":
    # In dev, run with: FLASK_APP=app.py flask run
    app.run(debug=True)
