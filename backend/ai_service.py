import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

MODEL_NAME = "gemini-2.0-flash-lite"

# Get API key from environment
API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configure API
def configure_genai(api_key: str):
    genai.configure(api_key=api_key)


# ---------------- IMAGE → CODE ----------------
async def extract_code_from_image(image_bytes: bytes, media_type: str = "image/png") -> str:
    import PIL.Image
    import io

    def _call():
        configure_genai(API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        img = PIL.Image.open(io.BytesIO(image_bytes))

        response = model.generate_content([
            img,
            "Extract ALL code from this screenshot exactly as written. Preserve formatting. Output ONLY raw code."
        ])

        return response.text.strip()

    return await asyncio.to_thread(_call)


# ---------------- DETECT LANGUAGE ----------------
async def detect_language(code: str) -> str:
    def _call():
        configure_genai(API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        response = model.generate_content(
            "What programming language is this? Reply ONLY with the language name.\n\n" + code[:1500]
        )

        return response.text.strip()

    return await asyncio.to_thread(_call)


# ---------------- HELPERS ----------------
def _numbered(code: str) -> str:
    return "\n".join(f"Line {i+1}: {l}" for i, l in enumerate(code.split("\n")))

def _extract(text: str, start: str, end: str) -> str:
    try:
        s = text.index(start) + len(start)
        e = text.index(end)
        return text[s:e].strip()
    except ValueError:
        return ""


# ---------------- PROMPTS ----------------
def prompt1(code: str, lang: str) -> str:
    return f"""
You are a senior {lang} engineer.

CODE:
{_numbered(code)}

LINE_EXPLANATIONS_START
Explain each line.
LINE_EXPLANATIONS_END

BUG_DETECTION_START
Find bugs if any.
BUG_DETECTION_END

CORRECTED_CODE_START
Give corrected code or CLEAN.
CORRECTED_CODE_END
"""

def prompt2(code: str, lang: str) -> str:
    return f"""
Analyze this {lang} code.

CODE:
{_numbered(code)}

DRY_RUN_START
Explain execution steps.
DRY_RUN_END

TIME_COMPLEXITY_START
Big-O explanation.
TIME_COMPLEXITY_END

SPACE_COMPLEXITY_START
Space explanation.
SPACE_COMPLEXITY_END

SUGGESTIONS_START
Give improvements.
SUGGESTIONS_END
"""


# ---------------- ANALYZE ----------------
def _sync_call(prompt: str) -> str:
    configure_genai(API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


async def analyze_code(code: str, language: str) -> dict:
    p1 = prompt1(code, language)
    p2 = prompt2(code, language)

    r1, r2 = await asyncio.gather(
        asyncio.to_thread(_sync_call, p1),
        asyncio.to_thread(_sync_call, p2),
    )

    return {
        "line_explanations": _extract(r1, "LINE_EXPLANATIONS_START", "LINE_EXPLANATIONS_END"),
        "bug_detection": _extract(r1, "BUG_DETECTION_START", "BUG_DETECTION_END"),
        "corrected_code": _extract(r1, "CORRECTED_CODE_START", "CORRECTED_CODE_END"),
        "dry_run": _extract(r2, "DRY_RUN_START", "DRY_RUN_END"),
        "time_complexity": _extract(r2, "TIME_COMPLEXITY_START", "TIME_COMPLEXITY_END"),
        "space_complexity": _extract(r2, "SPACE_COMPLEXITY_START", "SPACE_COMPLEXITY_END"),
        "suggestions": _extract(r2, "SUGGESTIONS_START", "SUGGESTIONS_END"),
    }


# ---------------- FOLLOW-UP ----------------
async def ask_followup(question: str, code: str, language: str) -> str:
    configure_genai(API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    context = f"""
You are a senior {language} engineer.

Code:
{code}

Question:
{question}
"""

    response = model.generate_content(context)
    return response.text.strip()