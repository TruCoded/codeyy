import os
import asyncio
import time
import re
from google import genai
from google.genai import types

_client = None

def get_client():
    global _client
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to backend/.env  →  GEMINI_API_KEY=AQ..."
        )
    if _client is None:
        _client = genai.Client(api_key=key)
    return _client

MODEL_NAME = "gemini-2.5-flash-lite"


# ── Vision OCR ─────────────────────────────────────────────────────────────

async def extract_code_from_image(image_bytes: bytes, media_type: str = "image/png") -> str:
    async def _call():
        client = get_client()
        prompt = "Extract ALL code from this screenshot exactly as written. Output ONLY raw code."
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=media_type)]
            )
            return response.text.strip()
        except: return ""
    return await _call()

# ── Language auto-detect ────────────────────────────────────────────────────

async def detect_language(code: str) -> str:
    async def _call():
        client = get_client()
        prompt = "What programming language is this? Reply with ONLY the language name.\n\n" + code[:1000]
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except: return "python"
    return await _call()


# ── MEGA PROMPT ─────────────────────────────────────────────────────────────

def get_mega_prompt(code: str, lang: str) -> str:
    lines = "\n".join(f"Line {i+1}: {l}" for i, l in enumerate(code.split("\n")))
    return f"""You are a senior {lang} engineer. Analyze this code.
Output your analysis using ONLY these tagged sections:

CODE:
{lines}

LINE_EXPLANATIONS_START
[For EVERY line: Line N: code | Explanation: 1 sentence]
LINE_EXPLANATIONS_END

BUG_DETECTION_START
[List bugs as BUG_1: Line N - ISSUE | FIX. If none, write BUG_NONE]
BUG_DETECTION_END

CORRECTED_CODE_START
[Full corrected code or CLEAN]
CORRECTED_CODE_END

DRY_RUN_START
[Trace execution: STEP_N: Line N | Action | Variables. End with RESULT: value]
DRY_RUN_END

TIME_COMPLEXITY_START
[Big-O and 1 sentence]
TIME_COMPLEXITY_END

SPACE_COMPLEXITY_START
[Big-O and 1 sentence]
SPACE_COMPLEXITY_END

SUGGESTIONS_START
[1-5 actionable tips]
SUGGESTIONS_END"""


def _extract(text: str, start: str, end: str) -> str:
    try:
        s = text.index(start) + len(start)
        e = text.index(end)
        return text[s:e].strip()
    except: return ""


async def analyze_code(code: str, language: str) -> dict:
    client = get_client()
    prompt = get_mega_prompt(code, language)
    
    # Single attempt with short retry
    for attempt in range(2):
        try:
            response = await asyncio.to_thread(client.models.generate_content, model=MODEL_NAME, contents=prompt)
            r = response.text
            return {
                "line_explanations": _extract(r, "LINE_EXPLANATIONS_START", "LINE_EXPLANATIONS_END"),
                "bug_detection":     _extract(r, "BUG_DETECTION_START",     "BUG_DETECTION_END"),
                "corrected_code":    _extract(r, "CORRECTED_CODE_START",    "CORRECTED_CODE_END"),
                "dry_run":           _extract(r, "DRY_RUN_START",           "DRY_RUN_END"),
                "time_complexity":   _extract(r, "TIME_COMPLEXITY_START",   "TIME_COMPLEXITY_END"),
                "space_complexity":  _extract(r, "SPACE_COMPLEXITY_START",  "SPACE_COMPLEXITY_END"),
                "suggestions":       _extract(r, "SUGGESTIONS_START",       "SUGGESTIONS_END"),
            }
        except Exception as e:
            if "429" in str(e) and attempt == 0:
                await asyncio.sleep(5)
            else:
                raise e


async def ask_followup(question: str, code: str, language: str, history: list) -> str:
    client = get_client()
    ctx = f"Senior {language} engineer. Code:\n{code}"
    chats = []
    for t in history[-4:]:
        if t.get("question"): chats.append({"role": "user", "parts": [{"text": t["question"]}]})
        if t.get("answer"): chats.append({"role": "model", "parts": [{"text": t["answer"]}]})
    
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=chats + [{"role": "user", "parts": [{"text": question}]}],
            config=types.GenerateContentConfig(system_instruction=ctx)
        )
        return res.text.strip()
    except: return "Service busy. Please try again."