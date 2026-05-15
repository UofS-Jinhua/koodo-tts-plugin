"""
FastAPI backend for the local audiobook reading app.
Provides endpoints for text upload, TTS synthesis, and audio streaming.
"""
import os
import io
import json
import logging
import asyncio
import threading
import time
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from text_processor import TextProcessor
from tts_engine import TTSEngine

# Configure logging
os.environ["PYTHONIOENCODING"] = "utf-8"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="AudioBook TTS", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
text_processor = TextProcessor()
tts_engine: Optional[TTSEngine] = None
_engine_lock = asyncio.Lock()
last_active_time = time.time()
active_streams = {}



def get_engine() -> TTSEngine:
    """Lazy-initialize the TTS engine on first use."""
    global tts_engine
    if tts_engine is None:
        logger.info("[App] Initializing TTS engine (first request)...")
        tts_engine = TTSEngine(character="kiana")
        logger.info("[App] TTS engine ready.")
    return tts_engine


async def idle_watchdog():
    """Watchdog task to monitor idle time and shut down the server."""
    global last_active_time
    
    # Wait a bit on startup to avoid premature shutdown
    await asyncio.sleep(20)
    
    while True:
        await asyncio.sleep(10)
        
        # 1. Check if launched by Koodo and Koodo is closed
        if os.environ.get("LAUNCHED_BY_KOODO") == "1":
            try:
                # Fast check using tasklist
                output = subprocess.check_output('tasklist', shell=True).decode('utf-8', errors='ignore').lower()
                if 'koodo' not in output:
                    logger.info("[App] Koodo Reader process is closed. Shutting down TTS server.")
                    os._exit(0)
            except Exception as e:
                logger.error(f"Error checking processes: {e}")
        
        # 2. Check 15min idle timeout
        if time.time() - last_active_time > 900:
            logger.info("[App] Server idle for 15 minutes. Shutting down to free memory.")
            os._exit(0)


@app.on_event("startup")
async def startup_event():
    """Start the idle watchdog on server startup."""
    asyncio.create_task(idle_watchdog())


# ── Request/Response Models ──────────────────────────────────────

class TextInput(BaseModel):
    text: str
    filename: str = "untitled.txt"

class CharacterInput(BaseModel):
    character: str


# ── API Endpoints ────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Check server and engine status."""
    engine = get_engine()
    return {
        "status": "ready",
        "character": engine.current_character,
        "characters": engine.get_characters(),
        "sample_rate": engine.sample_rate,
    }


@app.post("/api/upload")
async def upload_text(file: UploadFile = File(...)):
    """Upload a text file and split it into sentences."""
    content = await file.read()

    # Try multiple encodings
    text = None
    for encoding in ["utf-8", "gbk", "gb2312", "gb18030", "big5", "latin-1"]:
        try:
            text = content.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        raise HTTPException(status_code=400, detail="Unable to decode file encoding")

    doc_id = text_processor.load_text(text, filename=file.filename or "untitled.txt")
    doc = text_processor.get_document(doc_id)

    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "total_sentences": doc["total_sentences"],
        "sentences": doc["sentences"],
    }


@app.post("/api/text")
async def submit_text(input: TextInput):
    """Submit raw text and split into sentences."""
    doc_id = text_processor.load_text(input.text, filename=input.filename)
    doc = text_processor.get_document(doc_id)

    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "total_sentences": doc["total_sentences"],
        "sentences": doc["sentences"],
    }


@app.get("/api/synthesize/{doc_id}/{sentence_id}")
def synthesize_sentence(doc_id: str, sentence_id: int):
    """Synthesize a specific sentence to audio."""
    sentence = text_processor.get_sentence(doc_id, sentence_id)
    if sentence is None:
        raise HTTPException(status_code=404, detail="Sentence not found")

    # Fetch context (kept for caching footprint consistency)
    prev_sen = text_processor.get_sentence(doc_id, sentence_id - 1) if sentence_id > 0 else ""
    next_sen = text_processor.get_sentence(doc_id, sentence_id + 1) or ""

    engine = get_engine()
    
    global last_active_time
    last_active_time = time.time()

    try:
        wav_bytes = engine.synthesize(sentence, prev_text=prev_sen, next_text=next_sen)
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    if not wav_bytes:
        raise HTTPException(status_code=500, detail="Empty audio generated")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"inline; filename=sentence_{sentence_id}.wav",
        }
    )


@app.post("/api/synthesize_text")
def synthesize_text_endpoint(input: TextInput):
    """Directly synthesize raw text to audio."""
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    engine = get_engine()
    
    global last_active_time
    last_active_time = time.time()

    try:
        wav_bytes = engine.synthesize(input.text)
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    if not wav_bytes:
        raise HTTPException(status_code=500, detail="Empty audio generated")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=synthesis.wav",
        }
    )


@app.post("/api/prepare_stream")
async def prepare_stream(input: TextInput):
    """Store text and return a stream_id."""
    import uuid
    stream_id = str(uuid.uuid4())
    active_streams[stream_id] = input.text
    
    # Automatically clean up if not accessed within 60 seconds
    async def cleanup():
        await asyncio.sleep(60)
        if stream_id in active_streams:
            del active_streams[stream_id]
            logger.info(f"Cleaned up unaccessed stream {stream_id}")
            
    asyncio.create_task(cleanup())
    
    return {"stream_id": stream_id}


@app.get("/api/stream/{stream_id}")
def stream_text(stream_id: str):
    """Stream audio chunks continuously."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found or expired")
        
    text = active_streams.pop(stream_id)
    engine = get_engine()
    
    global last_active_time
    last_active_time = time.time()
    
    def audio_generator():
        try:
            for chunk in engine.synthesize_stream(text):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            
    return StreamingResponse(
        audio_generator(), 
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.post("/api/character")
async def change_character(input: CharacterInput):
    """Change the active TTS character."""
    engine = get_engine()
    try:
        engine.load_character(input.character)
        return {"status": "ok", "character": engine.current_character}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/characters")
async def list_characters():
    """List available characters."""
    engine = get_engine()
    return engine.get_characters()


# ── Static Files & SPA Fallback ─────────────────────────────────

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "Frontend not built yet"}, status_code=404)


# ── Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
