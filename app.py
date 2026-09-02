"""
FastAPI backend for the local audiobook reading app.
Provides endpoints for text upload, TTS synthesis, and audio streaming.
"""
import os
import logging
import asyncio
import threading
import time
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from text_processor import TextProcessor
from tts_engine import TTSEngine

# Configure logging
os.environ["PYTHONIOENCODING"] = "utf-8"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Global state
text_processor = TextProcessor()
tts_engine: Optional[TTSEngine] = None
# 必须是 threading.Lock：下面的合成端点是同步的 def，FastAPI 会把它们丢进线程池
# 并发执行，asyncio.Lock 在那里根本用不上。
_engine_lock = threading.Lock()
last_active_time = time.time()

WATCHDOG_INTERVAL = 30      # 秒
IDLE_TIMEOUT = 900          # 15 分钟
KOODO_MISS_THRESHOLD = 3    # 连续几次查不到 Koodo 才退出，避免 tasklist 偶发失败误杀

# Windows 下从无窗口进程调 tasklist 会闪黑框，用这个标志抑制
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def touch():
    """记录一次活动，用于 idle 计时。"""
    global last_active_time
    last_active_time = time.time()


def get_engine() -> TTSEngine:
    """Lazy-initialize the TTS engine on first use."""
    global tts_engine
    # 双重检查：Koodo 会一边播当前句一边预取下一句，两个线程可能同时走到这里。
    # 没有锁的话 ONNX 模型会被加载两份，内存翻倍甚至直接 OOM。
    if tts_engine is None:
        with _engine_lock:
            if tts_engine is None:
                logger.info("[App] Initializing TTS engine (first request)...")
                tts_engine = TTSEngine(character="kiana")
                logger.info("[App] TTS engine ready.")
    return tts_engine


def _koodo_is_running() -> Optional[bool]:
    """Koodo 是否还活着。查询失败返回 None（表示无法判断）。"""
    try:
        # 只查 Koodo 自己，不再每次拉取全量进程列表——CPU 推理本来就吃满核心，
        # 原来那种每 10 秒一次的全量 tasklist 会直接和 TTS 抢 CPU。
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Koodo*", "/NH"],
            capture_output=True, text=True, errors="ignore",
            creationflags=CREATE_NO_WINDOW,
        )
        return "koodo" in result.stdout.lower()
    except Exception as e:
        logger.error(f"Error checking processes: {e}")
        return None


async def idle_watchdog():
    """Watchdog task to monitor idle time and shut down the server."""
    # Wait a bit on startup to avoid premature shutdown
    await asyncio.sleep(20)

    koodo_misses = 0
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL)

        # 1. Check if launched by Koodo and Koodo is closed
        if os.environ.get("LAUNCHED_BY_KOODO") == "1":
            running = _koodo_is_running()
            if running is False:
                koodo_misses += 1
                if koodo_misses >= KOODO_MISS_THRESHOLD:
                    logger.info("[App] Koodo Reader process is closed. Shutting down TTS server.")
                    os._exit(0)
            elif running is True:
                koodo_misses = 0

        # 2. Check idle timeout
        if time.time() - last_active_time > IDLE_TIMEOUT:
            logger.info("[App] Server idle for 15 minutes. Shutting down to free memory.")
            os._exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the idle watchdog on server startup."""
    watchdog = asyncio.create_task(idle_watchdog())
    yield
    watchdog.cancel()


# Initialize app
app = FastAPI(title="AudioBook TTS", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    touch()
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
    touch()

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
    touch()

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


# 注：这里原本有一对 /api/prepare_stream + /api/stream 端点，但它们调用的
# engine.synthesize_stream() 根本不存在（genie-tts 只导出 tts / tts_async），
# 异常又被吞掉只写日志，客户端拿到的是 HTTP 200 + 空 body，看起来"成功"却没声音。
# 没有任何调用方（index.html 和两个 Koodo 插件都不用），故删除。
# 若日后要做真流式，入口是 genie.tts_async()，它返回裸 PCM 的 AsyncIterator[bytes]，
# 需要自己补 WAV 头。


@app.post("/api/character")
async def change_character(input: CharacterInput):
    """Change the active TTS character."""
    engine = get_engine()
    touch()
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
