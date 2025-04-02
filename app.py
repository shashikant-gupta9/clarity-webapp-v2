from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re
import os
from werkzeug.utils import secure_filename
from docx import Document
import fitz  # PyMuPDF

app = Flask(__name__, static_folder='static', template_folder='templates')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Extract text from URL
def extract_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        exclude_list = ['disclaimer', 'cookie', 'privacy policy']
        include_list = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        elements = [el for el in include_list if not any(keyword in el.get_text().lower() for keyword in exclude_list)]
        text = " ".join([el.get_text() for el in elements])
        text = re.sub(r'\n\s*\n', '\n', text)
        return text
    return ""

# File handling
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ""
    if ext == "txt":
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    elif ext == "docx":
        doc = Document(filepath)
        text = " ".join([p.text for p in doc.paragraphs])
    elif ext == "pdf":
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
    return text

# Split into chunks
def split_text_into_chunks(text, chunk_size=1024):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Summarize text
def summarize(text, chunk_size=1024, chunk_summary_size=128):
    chunks = split_text_into_chunks(text, chunk_size)
    summaries = []
    for chunk in chunks:
        size = chunk_summary_size if len(chunk) > chunk_summary_size else int(len(chunk)/2)
        try:
            summary = summarizer(chunk, min_length=1, max_length=size)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
    return " ".join(summaries)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/summarize', methods=['POST'])
def summarize_api():
    data = request.get_json()
    url = data.get("url")
    text = data.get("text")

    if url:
        text = extract_text(url)

    if not text:
        return jsonify({"error": "No input provided."}), 400

    summary = summarize(text)
    return jsonify({"summary": summary})

@app.route('/ask', methods=['POST'])
def ask_api():
    data = request.get_json()
    context = data.get("context")
    question = data.get("question")

    if not context or not question:
        return jsonify({"error": "Context and question required."}), 400

    try:
        answer = qa_pipeline(question=question, context=context)["answer"]
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_file(filepath)
        summary = summarize(text)
        return jsonify({"summary": summary})
    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
