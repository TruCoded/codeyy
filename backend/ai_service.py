import os
import asyncio
import time
import re
from google import genai
from google.genai import types

_clients = {}

def get_client(api_key: str = None):
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Please configure it in the application or backend/.env"
        )
    print(f"DEBUG: Using API key starting with {key[:10]}...")
    if key not in _clients:
        _clients[key] = genai.Client(api_key=key)
    return _clients[key]

MODEL_NAME = "gemini-2.5-flash"


# ── Vision OCR ─────────────────────────────────────────────────────────────

async def extract_code_from_image(image_bytes: bytes, media_type: str = "image/png", api_key: str = None) -> str:
    async def _call():
        client = get_client(api_key)
        prompt = "Extract ALL code from this screenshot exactly as written. Output ONLY raw code."
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=media_type)]
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error in extract_code_from_image: {e}")
            raise e
    return await _call()

# ── Language auto-detect ────────────────────────────────────────────────────

async def detect_language(code: str, api_key: str = None) -> str:
    async def _call():
        client = get_client(api_key)
        prompt = "What programming language is this? Reply with ONLY the language name.\n\n" + code[:1000]
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in detect_language: {e}")
            raise e
    return await _call()


# ── MEGA PROMPT ─────────────────────────────────────────────────────────────

def get_mega_prompt(code: str, lang: str) -> str:
    lines = "\n".join(f"Line {i+1}: {l}" for i, l in enumerate(code.split("\n")))
    engineer_role = f"senior {lang} engineer" if lang and lang.lower() != "auto" else "senior software engineer"
    return f"""You are a {engineer_role}. Analyze this code.
Output your analysis using ONLY these tagged sections:

CODE:
{lines}

DETECTED_LANGUAGE_START
[Output ONLY the lowercase identifier of the programming language, e.g. python, javascript, typescript, cpp, c, java, rust, go, html, css, ruby, php, etc.]
DETECTED_LANGUAGE_END

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

FLOWCHART_START
[Generate a Mermaid.js control flow diagram (graph TD) showing the execution flow of the code. Keep it simple and use valid Mermaid graph syntax.]
FLOWCHART_END

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


async def analyze_code(code: str, language: str, api_key: str = None) -> dict:
    client = get_client(api_key)
    prompt = get_mega_prompt(code, language)
    
    # Single attempt with short retry
    for attempt in range(2):
        try:
            response = await asyncio.to_thread(client.models.generate_content, model=MODEL_NAME, contents=prompt)
            r = response.text
            return {
                "detected_language": _extract(r, "DETECTED_LANGUAGE_START", "DETECTED_LANGUAGE_END"),
                "line_explanations": _extract(r, "LINE_EXPLANATIONS_START", "LINE_EXPLANATIONS_END"),
                "bug_detection":     _extract(r, "BUG_DETECTION_START",     "BUG_DETECTION_END"),
                "corrected_code":    _extract(r, "CORRECTED_CODE_START",    "CORRECTED_CODE_END"),
                "dry_run":           _extract(r, "DRY_RUN_START",           "DRY_RUN_END"),
                "flowchart":         _extract(r, "FLOWCHART_START",         "FLOWCHART_END"),
                "time_complexity":   _extract(r, "TIME_COMPLEXITY_START",   "TIME_COMPLEXITY_END"),
                "space_complexity":  _extract(r, "SPACE_COMPLEXITY_START",  "SPACE_COMPLEXITY_END"),
                "suggestions":       _extract(r, "SUGGESTIONS_START",       "SUGGESTIONS_END"),
            }
        except Exception as e:
            if "429" in str(e) and attempt == 0:
                await asyncio.sleep(5)
            else:
                raise e


async def refactor_code(code: str, language: str, target_complexity: str, api_key: str = None) -> dict:
    async def _call():
        client = get_client(api_key)
        prompt = f"""You are a senior {language} engineer. Refactor the following code to achieve a target complexity of: {target_complexity}.
If it's already at that complexity or cannot be refactored further, optimize it as much as possible while explaining why.

Output your response using ONLY these tagged sections:

REFACTORED_CODE_START
[Output ONLY the complete, optimized/refactored code. No Markdown code block fences.]
REFACTORED_CODE_END

EXPLANATION_START
[Provide a clear, brief explanation of the changes made and how they achieve the target complexity, or why they could not.]
EXPLANATION_END

Code to refactor:
{code}"""
        try:
            response = await asyncio.to_thread(client.models.generate_content, model=MODEL_NAME, contents=prompt)
            r = response.text
            return {
                "refactored_code": _extract(r, "REFACTORED_CODE_START", "REFACTORED_CODE_END"),
                "explanation": _extract(r, "EXPLANATION_START", "EXPLANATION_END")
            }
        except Exception as e:
            print(f"Error in refactor_code: {e}")
            raise e
    return await _call()


async def generate_comments_for_code(code: str, language: str, api_key: str = None) -> dict:
    async def _call():
        client = get_client(api_key)
        prompt = f"""You are a senior {language} engineer. Add detailed, professional docstrings, function headers, and clear inline comments explaining complex logic in the code. Keep the code structure and behavior exactly the same.

Output your response using ONLY these tagged sections:

COMMENTED_CODE_START
[Output ONLY the complete commented code. Do not use Markdown code block fences.]
COMMENTED_CODE_END

EXPLANATION_START
[Briefly summarize the comments and documentation standards added.]
EXPLANATION_END

Code:
{code}"""
        try:
            response = await asyncio.to_thread(client.models.generate_content, model=MODEL_NAME, contents=prompt)
            r = response.text
            return {
                "commented_code": _extract(r, "COMMENTED_CODE_START", "COMMENTED_CODE_END"),
                "explanation": _extract(r, "EXPLANATION_START", "EXPLANATION_END")
            }
        except Exception as e:
            print(f"Error in generate_comments_for_code: {e}")
            raise e
    return await _call()


async def ask_followup(question: str, code: str, language: str, history: list, api_key: str = None) -> str:
    client = get_client(api_key)
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
    except Exception as e:
        print(f"Error in ask_followup: {e}")
        raise e