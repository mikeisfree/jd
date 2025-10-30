# gemini_client.py
import os
import json
from google import genai
from typing import Dict

class GeminiError(Exception):
    pass

class GeminiClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        text = getattr(resp, "text", None) or getattr(resp, "response", None) or str(resp)
        return text

    def generate_json(self, prompt: str) -> Dict:
        text = self.generate_text(prompt)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise GeminiError("No JSON object found in model output.")
        json_text = text[start:end+1]
        try:
            data = json.loads(json_text)
        except Exception as e:
            raise GeminiError(f"JSON parse error: {e}\nRaw text: {text[:1000]}")
        return data
