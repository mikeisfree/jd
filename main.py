# main.py
import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape
from gemini_client import GeminiClient, GeminiError
from dotenv import load_dotenv

# Load .env
load_dotenv()

# FastAPI app
app = FastAPI(title="JD Generator")

# Gemini config from .env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

gemini = GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)

# Jinja2 setup
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"])
)
template = env.get_template("job_post.html")

# Pydantic models
class ChannelConfig(BaseModel):
    channel: str
    style: str
    candidate_profile: str
    language: str = "PL"

class GenerateRequest(BaseModel):
    job_title: str
    channels: List[ChannelConfig]

class TestRequest(BaseModel):
    job_title: str
    candidate_profile: str

class TestAnswersRequest(BaseModel):
    job_title: str
    candidate_profile: str
    questions: List[str]
    answers: List[str]

# Simple in-memory cache
CACHE_FILE = "cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        CACHE = json.load(f)
else:
    CACHE = {}

# ---------- Helper functions ----------
def build_prompt(job_title, channel, style, candidate_profile, language="PL") -> str:
    return f"""
You are an HR copywriting assistant. Produce only valid JSON that matches the following template:
{{
  "job_title": "{job_title}",
  "channel": "{channel}",
  "style": "{style}",
  "candidate_profile": "{candidate_profile}",
  "language": "{language}",
  "sections": {{
    "headline": "...",
    "subheadline": "...",
    "intro": "...",
    "responsibilities": ["..."],
    "requirements": ["..."],
    "nice_to_have": ["..."],
    "offer": ["..."],
    "why_us": "..."
  }}
}}

Rules:
- Output valid JSON only, no explanation.
- Tone/length must match channel:
  - social: punchy, emojis ok, shorter.
  - pracujpl: formal, bullet-style.
  - olx: short, direct, practical.
- style decides emphasis:
  - classic: responsibilities & requirements focus.
  - lifestyle: team, culture, flexibility.
  - growth: mentoring, learning, projects.
- candidate_profile modifies wording (students -> emphasize learning; experienced -> ownership; returners -> re-onboarding support).
- Keep each list of responsibilities/requirements to 3-6 items.
- Use Polish language unless language param says otherwise.
"""

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, ensure_ascii=False, indent=2)

def build_test_prompt(job_title, candidate_profile) -> str:
    return f"""
You are an HR assessment specialist. Generate exactly 5 technical/behavioral questions for a {job_title} position targeting {candidate_profile} candidates.

Output ONLY valid JSON in this format:
{{
  "questions": [
    "Question 1 text here?",
    "Question 2 text here?",
    "Question 3 text here?",
    "Question 4 text here?",
    "Question 5 text here?"
  ]
}}

Rules:
- Exactly 5 questions
- Mix of technical and behavioral questions
- Appropriate difficulty for {candidate_profile} level
- Questions should assess key competencies for {job_title}
- Use Polish language
"""

def build_scoring_prompt(job_title, candidate_profile, questions, answers) -> str:
    qa_pairs = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}\n" for i, (q, a) in enumerate(zip(questions, answers))])
    
    return f"""
You are an HR assessment specialist evaluating a candidate for {job_title} position ({candidate_profile} level).

Questions and Answers:
{qa_pairs}

Evaluate each answer and provide scores. Output ONLY valid JSON in this format:
{{
  "scores": [
    {{"question": 1, "score": 4, "feedback": "Brief feedback"}},
    {{"question": 2, "score": 3, "feedback": "Brief feedback"}},
    {{"question": 3, "score": 5, "feedback": "Brief feedback"}},
    {{"question": 4, "score": 2, "feedback": "Brief feedback"}},
    {{"question": 5, "score": 4, "feedback": "Brief feedback"}}
  ],
  "overall_score": 3.6,
  "grade": "B",
  "summary": "Overall assessment summary here"
}}

Rules:
- Score each answer 1-5 (1=poor, 5=excellent)
- Provide brief feedback for each
- Calculate overall_score as average
- Assign grade: A (4.5-5), B (3.5-4.4), C (2.5-3.4), D (1.5-2.4), F (1-1.4)
- Provide 2-3 sentence summary
- Use Polish language
"""

# ---------- Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JD Generator</title>
<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  padding: 2rem;
  color: #333;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
}

h1 {
  color: white;
  font-size: 2.5rem;
  margin-bottom: 2rem;
  text-align: center;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.card {
  background: rgba(30, 30, 46, 0.95);
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  margin-bottom: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

form {
  display: flex;
  gap: 2rem;
}

.form-left {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.form-right {
  flex: 1;
  display: flex;
  flex-direction: column;
}

@media (max-width: 768px) {
  form {
    flex-direction: column;
  }
}

label {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  font-weight: 500;
  color: #e5e7eb;
}

input[type="text"],
select {
  padding: 0.75rem;
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.2s;
  font-family: inherit;
  background: rgba(255, 255, 255, 0.05);
  color: #e5e7eb;
}

input[type="text"]:focus,
select:focus {
  outline: none;
  border-color: #667eea;
  background: rgba(255, 255, 255, 0.08);
}

select option {
  background: #1e1e2e;
  color: #e5e7eb;
}

button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 2rem;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  margin-top: 1rem;
}

button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
}

button:active {
  transform: translateY(0);
}

h2 {
  color: #e5e7eb;
  font-size: 1.5rem;
  margin-bottom: 1rem;
}

pre {
  background: rgba(0, 0, 0, 0.3);
  padding: 1.5rem;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 0.875rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  max-height: 400px;
  overflow-y: auto;
  color: #a5b4fc;
  word-wrap: break-word;
  white-space: pre-wrap;
}

.toolbar {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.toolbar button {
  background: rgba(255, 255, 255, 0.1);
  color: #e5e7eb;
  padding: 0.5rem 1rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  margin: 0;
}

.toolbar button:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: none;
  box-shadow: none;
}

.preview-container {
  position: relative;
  height: 800px;
}

.editable-preview {
  width: 100%;
  height: 100%;
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  background: white;
}

.refactor-btn {
  position: absolute;
  bottom: 2rem;
  right: 2rem;
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  color: white;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
  transition: transform 0.2s, box-shadow 0.2s;
  z-index: 10;
}

.refactor-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(245, 158, 11, 0.5);
}

.refactor-btn:active {
  transform: translateY(0);
}

.grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 2rem;
}

.grid .card {
  min-width: 0;
  overflow: hidden;
}

@media (max-width: 1024px) {
  .grid {
    grid-template-columns: 1fr;
  }
}

.loading {
  display: none;
  text-align: center;
  color: #a5b4fc;
  font-weight: 600;
  margin-top: 1rem;
}

.loading.active {
  display: block;
}

.test-btn {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  margin-top: 1rem;
}

.test-btn:hover {
  box-shadow: 0 10px 20px rgba(16, 185, 129, 0.4);
}

.modal {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  z-index: 1000;
  overflow-y: auto;
}

.modal.active {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.modal-content {
  background: rgba(30, 30, 46, 0.98);
  border-radius: 12px;
  padding: 2rem;
  max-width: 700px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.modal-header h2 {
  margin: 0;
}

.close-btn {
  background: transparent;
  color: #e5e7eb;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  margin: 0;
  width: auto;
}

.close-btn:hover {
  color: #fff;
  transform: none;
  box-shadow: none;
}

.question-block {
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.question-block h4 {
  color: #a5b4fc;
  margin-bottom: 0.75rem;
}

.question-block textarea {
  width: 100%;
  min-height: 80px;
  padding: 0.75rem;
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  color: #e5e7eb;
  font-family: inherit;
  font-size: 1rem;
  resize: vertical;
}

.question-block textarea:focus {
  outline: none;
  border-color: #667eea;
}

.score-result {
  padding: 1rem;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 8px;
  margin-bottom: 1rem;
}

.score-item {
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
}

.score-item h5 {
  color: #a5b4fc;
  margin-bottom: 0.5rem;
}

.score-badge {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  font-weight: 600;
  margin-left: 0.5rem;
}

.grade-A { background: #10b981; color: white; }
.grade-B { background: #3b82f6; color: white; }
.grade-C { background: #f59e0b; color: white; }
.grade-D { background: #ef4444; color: white; }
.grade-F { background: #991b1b; color: white; }
</style>
</head>
<body>
<div class="container">
  <h1>🚀 JD Generator</h1>
  
  <div class="card">
    <form id="generateForm">
      <div class="form-left">
        <label>
          Job Title
          <input type="text" name="job_title" value="Junior Cybersecurity Engineer" required />
        </label>

        <label>
          Channel
          <select name="channel">
            <option value="social">Social Media</option>
            <option value="pracujpl">Pracuj.pl</option>
            <option value="olx">OLX</option>
          </select>
        </label>

        <label>
          Style
          <select name="style">
            <option value="classic">Classic</option>
            <option value="lifestyle">Lifestyle</option>
            <option value="growth">Growth</option>
          </select>
        </label>

        <label>
          Candidate Profile
          <select name="candidate_profile">
            <option value="students">Students</option>
            <option value="experienced">Experienced</option>
            <option value="returners">Returners</option>
          </select>
        </label>

        <button type="submit">Generate Job Post</button>
        <div class="loading" id="loading">Generating...</div>
      </div>

      <div class="form-right">
        <fieldset style="border: 1px solid rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; height: 100%;">
          <legend style="color: #e5e7eb; font-weight: 600; padding: 0 0.5rem;">Include Sections</legend>
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <label style="flex-direction: row; gap: 0.5rem; align-items: center; cursor: pointer;">
              <input type="checkbox" name="include_headline" checked style="width: auto; cursor: pointer;" />
              <span>Headline & Subheadline</span>
            </label>
            <label style="flex-direction: row; gap: 0.5rem; align-items: center; cursor: pointer;">
              <input type="checkbox" name="include_intro" checked style="width: auto; cursor: pointer;" />
              <span>Introduction</span>
            </label>
            <label style="flex-direction: row; gap: 0.5rem; align-items: center; cursor: pointer;">
              <input type="checkbox" name="include_responsibilities" checked style="width: auto; cursor: pointer;" />
              <span>Responsibilities</span>
            </label>
            <label style="flex-direction: row; gap: 0.5rem; align-items: center; cursor: pointer;">
              <input type="checkbox" name="include_requirements" checked style="width: auto; cursor: pointer;" />
              <span>Requirements</span>
            </label>
            <label style="flex-direction: row; gap: 0.5rem; align-items: center; cursor: pointer;">
              <input type="checkbox" name="include_nice_to_have" checked style="width: auto; cursor: pointer;" />
              <span>Nice to Have</span>
            </label>
            <label style="flex-direction: row; gap: 0.5rem; align-items: center; cursor: pointer;">
              <input type="checkbox" name="include_offer" checked style="width: auto; cursor: pointer;" />
              <span>What We Offer</span>
            </label>
            <label style="flex-direction: row; gap: 0.5rem; align-items: center; cursor: pointer;">
              <input type="checkbox" name="include_why_us" checked style="width: auto; cursor: pointer;" />
              <span>Why Join Us</span>
            </label>
          </div>
        </fieldset>
        <button type="button" id="generateTestBtn" class="test-btn">📝 Generate Test</button>
      </div>
    </form>
  </div>

  <div id="testModal" class="modal">
    <div class="modal-content">
      <div class="modal-header">
        <h2 id="modalTitle">Candidate Assessment Test</h2>
        <button class="close-btn" onclick="closeTestModal()">×</button>
      </div>
      <div id="modalBody"></div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>JSON Output</h2>
      <pre id="result">Click "Generate" to see results...</pre>
    </div>
    
    <div class="card">
      <h2>HTML Preview</h2>
      <div class="toolbar" id="toolbar" style="display: none;">
        <button onclick="document.execCommand('bold', false, null)" title="Bold">B</button>
        <button onclick="document.execCommand('italic', false, null)" title="Italic">I</button>
        <button onclick="document.execCommand('underline', false, null)" title="Underline">U</button>
        <button onclick="document.execCommand('insertUnorderedList', false, null)" title="Bullet List">•</button>
      </div>
      <div class="preview-container">
        <iframe id="editablePreview" class="editable-preview"></iframe>
        <button id="refactorBtn" class="refactor-btn" style="display: none;">
          ✨ Refactor Design
        </button>
      </div>
    </div>
  </div>
</div>

<script>
const form = document.getElementById('generateForm');
const result = document.getElementById('result');
const editablePreview = document.getElementById('editablePreview');
const refactorBtn = document.getElementById('refactorBtn');
const toolbar = document.getElementById('toolbar');
const loading = document.getElementById('loading');

let currentJsonData = null;
let currentChannel = null;

function loadHTMLIntoIframe(html) {
    const iframeDoc = editablePreview.contentDocument || editablePreview.contentWindow.document;
    iframeDoc.open();
    iframeDoc.write(html);
    iframeDoc.close();
    
    // Make iframe content editable
    setTimeout(() => {
        iframeDoc.body.contentEditable = true;
        iframeDoc.body.style.outline = 'none';
    }, 100);
}

function getIncludedSections() {
    return {
        headline: form.include_headline.checked,
        intro: form.include_intro.checked,
        responsibilities: form.include_responsibilities.checked,
        requirements: form.include_requirements.checked,
        nice_to_have: form.include_nice_to_have.checked,
        offer: form.include_offer.checked,
        why_us: form.include_why_us.checked
    };
}

function filterHTMLBySections(html, sections) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    
    if (!sections.headline) {
        const header = doc.querySelector('header');
        if (header) header.remove();
    }
    
    const allSections = doc.querySelectorAll('section');
    allSections.forEach(section => {
        const h3 = section.querySelector('h3');
        const heading = h3 ? h3.textContent.trim() : '';
        
        if (!sections.intro && !h3 && section.querySelector('p')) {
            section.remove();
        } else if (!sections.responsibilities && heading.includes('zadania')) {
            section.remove();
        } else if (!sections.requirements && heading.includes('Wymagania')) {
            section.remove();
        } else if (!sections.nice_to_have && heading.includes('Mile widziane')) {
            section.remove();
        } else if (!sections.offer && heading.includes('oferujemy')) {
            section.remove();
        } else if (!sections.why_us && heading.includes('warto')) {
            section.remove();
        }
    });
    
    return doc.documentElement.outerHTML;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    loading.classList.add('active');
    
    const data = {
        job_title: form.job_title.value,
        channels: [{
            channel: form.channel.value,
            style: form.style.value,
            candidate_profile: form.candidate_profile.value
        }]
    };
    
    try {
        const res = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const json = await res.json();
        currentJsonData = json;
        currentChannel = form.channel.value;
        
        result.textContent = JSON.stringify(json, null, 2);
        
        const sections = getIncludedSections();
        const filteredHTML = filterHTMLBySections(json[currentChannel].html, sections);
        loadHTMLIntoIframe(filteredHTML);
        
        toolbar.style.display = 'flex';
        refactorBtn.style.display = 'block';
    } catch (error) {
        result.textContent = 'Error: ' + error.message;
    } finally {
        loading.classList.remove('active');
    }
});

refactorBtn.addEventListener('click', async () => {
    if (!currentJsonData || !currentChannel) return;
    
    refactorBtn.textContent = '⏳ Refactoring...';
    refactorBtn.disabled = true;
    
    try {
        const data = {
            job_title: form.job_title.value,
            channels: [{
                channel: currentChannel,
                style: form.style.value,
                candidate_profile: form.candidate_profile.value
            }]
        };
        
        const res = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const json = await res.json();
        currentJsonData = json;
        
        result.textContent = JSON.stringify(json, null, 2);
        
        const sections = getIncludedSections();
        const filteredHTML = filterHTMLBySections(json[currentChannel].html, sections);
        loadHTMLIntoIframe(filteredHTML);
        
    } catch (error) {
        alert('Error refactoring: ' + error.message);
    } finally {
        refactorBtn.textContent = '✨ Refactor Design';
        refactorBtn.disabled = false;
    }
});

// Toolbar functionality for iframe
toolbar.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') {
        const iframeDoc = editablePreview.contentDocument || editablePreview.contentWindow.document;
        iframeDoc.body.focus();
    }
});

// Test generation functionality
const generateTestBtn = document.getElementById('generateTestBtn');
const testModal = document.getElementById('testModal');
const modalBody = document.getElementById('modalBody');
let currentTest = null;

generateTestBtn.addEventListener('click', async () => {
    const jobTitle = form.job_title.value;
    const candidateProfile = form.candidate_profile.value;
    
    generateTestBtn.textContent = '⏳ Generating Test...';
    generateTestBtn.disabled = true;
    
    try {
        const res = await fetch('/generate-test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                job_title: jobTitle,
                candidate_profile: candidateProfile
            })
        });
        const testData = await res.json();
        currentTest = testData;
        
        showTestQuestions(testData.questions, jobTitle);
    } catch (error) {
        alert('Error generating test: ' + error.message);
    } finally {
        generateTestBtn.textContent = '📝 Generate Test';
        generateTestBtn.disabled = false;
    }
});

function showTestQuestions(questions, jobTitle) {
    let html = `<p style="color: #a5b4fc; margin-bottom: 1.5rem;">Answer the following questions for the ${jobTitle} position:</p>`;
    
    questions.forEach((q, i) => {
        html += `
            <div class="question-block">
                <h4>Question ${i + 1}</h4>
                <p style="color: #e5e7eb; margin-bottom: 0.75rem;">${q}</p>
                <textarea id="answer_${i}" placeholder="Your answer..."></textarea>
            </div>
        `;
    });
    
    html += `<button onclick="submitTest()" style="width: 100%;">Submit Test</button>`;
    
    modalBody.innerHTML = html;
    testModal.classList.add('active');
}

async function submitTest() {
    const answers = [];
    const questions = currentTest.questions;
    
    for (let i = 0; i < questions.length; i++) {
        const answer = document.getElementById(`answer_${i}`).value.trim();
        if (!answer) {
            alert(`Please answer question ${i + 1}`);
            return;
        }
        answers.push(answer);
    }
    
    modalBody.innerHTML = '<p style="text-align: center; color: #a5b4fc;">⏳ Scoring your answers...</p>';
    
    try {
        const res = await fetch('/score-test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                job_title: form.job_title.value,
                candidate_profile: form.candidate_profile.value,
                questions: questions,
                answers: answers
            })
        });
        const scoreData = await res.json();
        
        showTestResults(scoreData, questions, answers);
    } catch (error) {
        alert('Error scoring test: ' + error.message);
        closeTestModal();
    }
}

function showTestResults(scoreData, questions, answers) {
    let html = `
        <div class="score-result">
            <h3 style="color: #10b981; margin-bottom: 0.5rem;">
                Overall Score: ${scoreData.overall_score.toFixed(1)}/5.0
                <span class="score-badge grade-${scoreData.grade}">Grade: ${scoreData.grade}</span>
            </h3>
            <p style="color: #e5e7eb;">${scoreData.summary}</p>
        </div>
    `;
    
    scoreData.scores.forEach((score, i) => {
        html += `
            <div class="score-item">
                <h5>Question ${score.question}: Score ${score.score}/5</h5>
                <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 0.5rem;">${questions[i]}</p>
                <p style="color: #e5e7eb; font-size: 0.9rem; margin-bottom: 0.5rem;"><strong>Your answer:</strong> ${answers[i]}</p>
                <p style="color: #a5b4fc; font-size: 0.9rem;"><strong>Feedback:</strong> ${score.feedback}</p>
            </div>
        `;
    });
    
    html += `<button onclick="closeTestModal()" style="width: 100%; margin-top: 1rem;">Close</button>`;
    
    modalBody.innerHTML = html;
}

function closeTestModal() {
    testModal.classList.remove('active');
    modalBody.innerHTML = '';
    currentTest = null;
}

// Close modal on outside click
testModal.addEventListener('click', (e) => {
    if (e.target === testModal) {
        closeTestModal();
    }
});
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)

@app.post("/generate")
async def generate(payload: GenerateRequest):
    results = {}
    for cfg in payload.channels:
        cache_key = f"{payload.job_title}_{cfg.channel}_{cfg.style}_{cfg.candidate_profile}_{cfg.language}"
        if cache_key in CACHE:
            generated_json = CACHE[cache_key]
        else:
            prompt = build_prompt(
                job_title=payload.job_title,
                channel=cfg.channel,
                style=cfg.style,
                candidate_profile=cfg.candidate_profile,
                language=cfg.language
            )
            try:
                generated_json = gemini.generate_json(prompt)
            except GeminiError as e:
                raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

            CACHE[cache_key] = generated_json
            save_cache()

        html = template.render(data=generated_json)
        results[cfg.channel] = {
            "json": generated_json,
            "html": html
        }
    return results

@app.post("/generate-test")
async def generate_test(payload: TestRequest):
    prompt = build_test_prompt(payload.job_title, payload.candidate_profile)
    try:
        test_data = gemini.generate_json(prompt)
        return test_data
    except GeminiError as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

@app.post("/score-test")
async def score_test(payload: TestAnswersRequest):
    prompt = build_scoring_prompt(
        payload.job_title,
        payload.candidate_profile,
        payload.questions,
        payload.answers
    )
    try:
        scoring_data = gemini.generate_json(prompt)
        return scoring_data
    except GeminiError as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")
