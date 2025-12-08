# app/model.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqModelManager:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env")

        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"

    def predict_both(self, text, num_words=3, num_sentences=3):
        """
        One SINGLE API call that returns BOTH:
         - next words
         - next sentences
        """

        prompt = f"""
You are a predictive text model.
Given input text below, generate BOTH:

1) {num_words} likely next single words
2) {num_sentences} possible next sentences

Rules:
- Do NOT repeat the input text.
- Do NOT produce duplicate items.
- Output strictly in JSON object like this:
{{
  "words": [...],
  "sentences": [...]
}}

Input text: "{text}"
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.8,
                top_p=0.9,
            )
        except Exception as e:
            print("Model request failed:", e)
            return [], []

        import json, re
        # attempt to extract text content from response safely
        try:
            content = resp.choices[0].message.content
        except Exception:
            try:
                # older SDK shapes
                content = getattr(resp.choices[0], 'text', '')
            except Exception:
                content = ""

        if not content:
            return [], []

        # Some models include backticks or explanation; try to find a JSON object inside
        try:
            # find the first '{' and last '}' and parse between them
            m = re.search(r"\{.*\}", content, flags=re.S)
            if m:
                blob = m.group(0)
            else:
                # fallback: try to parse the whole content
                blob = content
            data = json.loads(blob)
            words = data.get("words") or data.get("word") or []
            sentences = data.get("sentences") or data.get("sentence") or []
            # ensure lists
            if not isinstance(words, list):
                words = [str(words)]
            if not isinstance(sentences, list):
                sentences = [str(sentences)]
            return words, sentences
        except Exception as e:
            print("Failed to parse model output as JSON; returning empty lists. Error:", e)
            # as last resort try to split content into lines and return top tokens
            try:
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                # guess words from first line tokens
                words = []
                sentences = []
                if lines:
                    # take first line tokens as words
                    first = lines[0]
                    words = [t for t in re.split(r"\s+|,", first) if t][:num_words]
                    # take next non-empty lines as sentences
                    for ln in lines[1:1+num_sentences]:
                        sentences.append(ln)
                return words, sentences
            except Exception:
                return [], []
        

_model_manager = None

def get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = GroqModelManager()
    return _model_manager
