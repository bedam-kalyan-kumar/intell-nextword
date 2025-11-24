# app/main.py
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from app.model import get_model_manager
from app.spellchecker import SpellChecker
from app.utils import top_k_unique
import re
from collections import OrderedDict


app = FastAPI(title="Intell Next-Word & Next-Sentence API (Lang-aware)")
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
def root():
    return FileResponse("web/index.html")

# singletons
model_manager = get_model_manager()
spell = SpellChecker()

class PredictResponse(BaseModel):
    original: str
    corrected: str
    word_candidates: List[str]
    sentence_candidates: List[str]

@app.api_route("/predict", methods=["GET","POST"], response_model=PredictResponse)
def predict(
    text: str = Query(..., min_length=1, description="Input text"),
    lang: str = Query("en", description="Language code, e.g., en, hi, te"),
    apply_spellcheck: bool = Query(True),
    word_max_tokens: int = Query(3),
    sentence_max_tokens: int = Query(100),
    num_word: int = Query(3),
    num_sentence: int = Query(3),
    do_sample: bool = Query(False)
):
    original = text
    # normalize lang to basic code
    lang = (lang or "en").lower()
    if apply_spellcheck:
        corrected = spell.correct_text(text, lang=lang)
    else:
        corrected = text

    # ----------------------------
    # Generation: ask model for word & sentence continuations
    # ----------------------------
    # Note: GPT-2 Large / your model implementation expects no 'lang' parameter.
    # We pass sampling options via do_sample; model.predict_next may also accept temperature/top_k/top_p.
    word_gen = model_manager.predict_next(
        corrected,
        max_new_tokens=word_max_tokens,
        do_sample=do_sample,
        num_return_sequences=num_word
    )

    sentence_gen = model_manager.predict_next(
        corrected,
        max_new_tokens=sentence_max_tokens,
        do_sample=do_sample,
        num_return_sequences=num_sentence
    )

    # ----------------------------
    # Post-processing / cleaning
    # ----------------------------
    def _normalize_whitespace(s: str) -> str:
        return re.sub(r'\s+', ' ', s).strip()

    def _is_repetitive(s: str) -> bool:
        toks = s.split()
        if len(toks) <= 1:
            return False
        # repeated token check: if one token appears >55% of the tokens it's repetitive
        for token in set(toks):
            if toks.count(token) > len(toks) * 0.55:
                return True
        # repeated substring check (e.g., repeated phrases)
        if re.search(r'(.{5,30}).*\1', s):
            return True
        return False

    def clean_candidates(cands, prompt_text):
        seen = OrderedDict()
        for c in cands:
            if not c:
                continue
            c2 = _normalize_whitespace(c)
            # drop if identical to prompt or empty
            if not c2 or c2.lower() == prompt_text.lower():
                continue
            # drop obviously repetitive/broken outputs
            if _is_repetitive(c2.lower()):
                continue
            # strip undesired leading/trailing punctuation/quotes
            c2 = c2.strip(' "\'`.,;:()[]{}')
            # collapse runaway repetitions like "do his job. do his job." -> keep first
            c2 = re.sub(r'(\b.+?\b)(?:\s*\1)+', r'\1', c2)
            if c2.lower() not in seen:
                seen[c2.lower()] = c2
        return list(seen.values())

    # Clean raw outputs
    word_candidates_clean = clean_candidates(word_gen, corrected)
    sentence_candidates_clean = clean_candidates(sentence_gen, corrected)

    # Optional grammar polishing (LanguageTool). Disabled by default to avoid extra deps/latency.
    USE_LANGTOOL = False
    if USE_LANGTOOL:
        try:
            import language_tool_python
            tool = language_tool_python.LanguageTool('en-US')  # change if needed
            def grammar_fix_list(lst):
                out = []
                for s in lst:
                    try:
                        fixed = tool.correct(s)
                    except Exception:
                        fixed = s
                    out.append(_normalize_whitespace(fixed))
                return out
            word_candidates_clean = grammar_fix_list(word_candidates_clean)
            sentence_candidates_clean = grammar_fix_list(sentence_candidates_clean)
        except Exception as e:
            print("LanguageTool error (ignored):", e)

    # Ensure we return exactly num_word / num_sentence items (pad or truncate)
    def pad_or_truncate(arr, k):
        if not arr:
            return []
        if len(arr) >= k:
            return arr[:k]
        fallbacks = []
        # produce fallbacks from tokens of existing candidates
        for part in arr:
            for tok in part.split():
                if tok not in arr and tok not in fallbacks:
                    fallbacks.append(tok)
                if len(arr) + len(fallbacks) >= k:
                    break
            if len(arr) + len(fallbacks) >= k:
                break
        return arr + fallbacks[:max(0, k - len(arr))]

    word_candidates = pad_or_truncate(word_candidates_clean, num_word)
    sentence_candidates = pad_or_truncate(sentence_candidates_clean, num_sentence)

    # As a final safety, ensure uniqueness and reasonable trimming (optional)
    word_candidates = top_k_unique([w.strip() for w in word_candidates], k=num_word)
    sentence_candidates = top_k_unique([s.strip() for s in sentence_candidates], k=num_sentence)

    return {
        "original": original,
        "corrected": corrected,
        "word_candidates": word_candidates,
        "sentence_candidates": sentence_candidates
    }
