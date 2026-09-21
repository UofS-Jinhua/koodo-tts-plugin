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

# 排查"服务端全 200 但播放突然中断"这类问题时，控制台窗口是唯一的证据来源：
# 没有时间戳就看不出请求之间隔了多久（例如空闲看门狗是不是刚好在那个点触发了
# 关闭），而且 start_server.bat 从不把输出重定向到文件——Koodo 拉起的那个窗口
# 一关，连这唯一的证据也没了。这里补上时间戳，并额外落一份滚动日志文件，
# 这样窗口关掉之后还能翻 logs/server.log。
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

# 5MB x 3 份滚动，够存好几天的会话，不会无限增长
from logging.handlers import RotatingFileHandler
_file_handler = RotatingFileHandler(
    LOG_DIR / "server.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_log_formatter)

# force=True 是必须的：上面 `from tts_engine import TTSEngine` 这条 import 链
# （genie-tts / transformers 等）会在导入期间自己给 root logger 装一个默认
# StreamHandler。logging.basicConfig() 发现 root logger 已经有 handler 就
# 直接静默跳过——不加 force=True 的话，下面这行等于什么也没做，我们自己配的
# 文件 handler 压根不会生效（实测过：没有 force=True 时 server.log 一直是 0 字节）。
logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler], force=True)
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
    # uvicorn 自己的 "uvicorn.access" / "uvicorn.error" logger 是独立的，
    # propagate=False，不会冒泡到 root——就是产生 `"POST ... " 200 OK` 那些行
    # 的 logger，而且它们默认没有时间戳、也不会落进我们上面配的文件 handler。
    # uvicorn 会在启动过程中用自己的 dictConfig 覆盖这两个 logger，所以不能在
    # 模块顶层去接管它们（会被 uvicorn 后来的配置盖掉）；lifespan 的启动阶段
    # 保证运行在 uvicorn 那次 configure_logging() 之后，这里接管才稳。
    for _name in ("uvicorn.access", "uvicorn.error"):
        _ulogger = logging.getLogger(_name)
        _ulogger.handlers = [_console_handler, _file_handler]
        _ulogger.propagate = False

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
