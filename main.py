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
app = FastAPI(title="JD Generator API")

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
    "why_us": "...",
    "cta": "..."
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

# ---------- Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>JD Generator</title>
</head>
<body>
<h1>JD Generator</h1>
<form id="generateForm">
  <label>Job Title: <input type="text" name="job_title" value="Junior Cybersecurity Engineer" /></label><br><br>

  <label>Channel:
    <select name="channel">
      <option value="social">Social</option>
      <option value="pracujpl">Pracuj.pl</option>
      <option value="olx">OLX</option>
    </select>
  </label><br><br>

  <label>Style:
    <select name="style">
      <option value="classic">Classic</option>
      <option value="lifestyle">Lifestyle</option>
      <option value="growth">Growth</option>
    </select>
  </label><br><br>

  <label>Candidate Profile:
    <select name="candidate_profile">
      <option value="students">Students</option>
      <option value="experienced">Experienced</option>
      <option value="returners">Returners</option>
    </select>
  </label><br><br>

  <button type="submit">Generate</button>
</form>

<h2>JSON Output</h2>
<pre id="result" style="background:#f0f0f0; padding:10px;"></pre>

<h2>HTML Preview</h2>
<iframe id="htmlPreview" style="width:100%; height:600px; border:1px solid #ccc;"></iframe>

<script>
const form = document.getElementById('generateForm');
const result = document.getElementById('result');
const iframe = document.getElementById('htmlPreview');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = {
        job_title: form.job_title.value,
        channels: [{
            channel: form.channel.value,
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
    result.textContent = JSON.stringify(json, null, 2);

    iframe.srcdoc = json[form.channel.value].html;
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
