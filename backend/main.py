import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
print(f"DEBUG: Loaded API key from {env_path.absolute()} -> prefix: {os.getenv('GEMINI_API_KEY')[:10] if os.getenv('GEMINI_API_KEY') else 'NONE'}")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import base64
import traceback

from ai_service import analyze_code, ask_followup, extract_code_from_image

app = FastAPI(title="Codeyy (Gemini)", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ─────────────────────────────────────────────────────────
FRONTEND = Path(__file__).parent.parent / "frontend"

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(FRONTEND / "index.html")

@app.get("/index.html", response_class=HTMLResponse)
async def index():
    return FileResponse(FRONTEND / "index.html")

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


# ── Models ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    code:             Optional[str] = None
    image_base64:     Optional[str] = None
    image_media_type: Optional[str] = "image/png"
    language:         str           = "python"

class AskRequest(BaseModel):
    question:             str
    code:                 str
    language:             str
    conversation_history: list = []

class KeyConfigRequest(BaseModel):
    api_key: str


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    header_key = request.headers.get("X-Gemini-API-Key", "")
    key = (header_key or os.getenv("GEMINI_API_KEY", "")).strip()
    ok  = bool(key and len(key) > 10 and (key.startswith("AIzaSy") or key.startswith("AQ.")))
    return JSONResponse({
        "ok":         ok,
        "key_prefix": key[:12] + "…" if (ok and len(key) >= 12) else "NOT SET",
        "model":      "gemini-1.5-flash",
        "version":    "4.1.0",
    })

@app.get("/api/key")
async def get_api_key(request: Request):
    client_host = request.client.host if request.client else ""
    is_local = client_host in ("127.0.0.1", "localhost", "::1")
    is_render = os.getenv("RENDER") == "true"
    
    if is_render or not is_local:
        return JSONResponse({"api_key": ""})
        
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or not (key.startswith("AIzaSy") or key.startswith("AQ.")):
        return JSONResponse({"api_key": ""})
    return JSONResponse({"api_key": key})

@app.post("/api/key")
async def save_api_key(req: KeyConfigRequest, request: Request):
    client_host = request.client.host if request.client else ""
    is_local = client_host in ("127.0.0.1", "localhost", "::1")
    is_render = os.getenv("RENDER") == "true"
    
    if is_render or not is_local:
        return JSONResponse({"status": "error", "message": "API key modification not allowed on public deployments"})
        
    new_key = req.api_key.strip()
    env_path = Path(__file__).parent / ".env"
    
    content = ""
    key_written = False
    
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith("GEMINI_API_KEY="):
                new_lines.append(f"GEMINI_API_KEY={new_key}")
                key_written = True
            else:
                new_lines.append(line)
        if not key_written:
            new_lines.append(f"GEMINI_API_KEY={new_key}")
        content = "\n".join(new_lines) + "\n"
    else:
        content = f"GEMINI_API_KEY={new_key}\n"
        
    env_path.write_text(content, encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = new_key
    
    return JSONResponse({"status": "ok", "message": "API key saved successfully"})


# ── /analyze ────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, request: Request):
    code = (req.code or "").strip()
    user_key = request.headers.get("X-Gemini-API-Key", "")

    if not code and req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64)
            code = await extract_code_from_image(raw, req.image_media_type or "image/png", api_key=user_key)
            if not code.strip():
                raise HTTPException(400, "Could not extract code. Use a high-contrast screenshot.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Image extraction failed: {e}")

    if not code:
        raise HTTPException(400, "No code provided.")

    try:
        result = await analyze_code(code, req.language, api_key=user_key)
        result["extracted_code"] = code if req.image_base64 else None
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Analysis error: {e}")


# ── /ask ────────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "Empty question.")
    if not req.code.strip():
        raise HTTPException(400, "No code context.")
    user_key = request.headers.get("X-Gemini-API-Key", "")
    try:
        answer = await ask_followup(
            req.question.strip(),
            req.code.strip(),
            req.language,
            req.conversation_history,
            api_key=user_key,
        )
        return JSONResponse({"answer": answer})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Q&A error: {e}")
