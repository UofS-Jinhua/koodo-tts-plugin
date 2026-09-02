"""
TTS Engine wrapper for Genie-TTS (GPT-SoVITS ONNX Inference).
Provides a clean interface for text-to-speech synthesis.
"""
import os
import io
import re
import wave
import logging
import tempfile
import threading
import hashlib
import genie_tts as genie
import mixed_g2p
from text_processor import TextProcessor

import json

logger = logging.getLogger(__name__)

# genie-tts 的中文 G2P 会把句子里夹带的英文整段删掉（详见 mixed_g2p 的模块说明），
# 必须在加载角色（会给参考文本做 G2P）之前把补丁打上。
mixed_g2p.apply()

# 获取当前文件所在目录的绝对路径，用于计算相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_characters_config():
    """Load characters configuration from json file."""
    config_path = os.path.join(BASE_DIR, "characters.json")
    if not os.path.exists(config_path):
        # Create an empty template if not exists
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Dynamically loaded configurations
CHARACTERS_CONFIG = load_characters_config()

# 片段之间插入的短停顿，避免拼接处听起来生硬
INTER_SEGMENT_PAUSE_SEC = 0.12
# 至少要有一个能发音的字符，否则送进 TTS 只会得到 0 字节文件
# (\w 在 Python 3 的 str 模式下已经涵盖汉字、假名等 Unicode 字母)
SPEAKABLE = re.compile(r'\w')

class TTSEngine:
    """Wraps Genie-TTS for GPT-SoVITS inference."""

    def __init__(self, character: str = "kiana"):
        self._loaded_characters = set()
        self._current_character = None
        self._sample_rate = 32000  # GPT-SoVITS default
        self._text_processor = TextProcessor()
        self._lock = threading.Lock()
        
        # Simple LRU Cache to handle Koodo timeout retries
        self._cache = {}
        self._cache_keys = []
        self._cache_max_size = 30

        logger.info(f"[TTS] Initializing with character: {character}")
        self.load_character(character)

    def load_character(self, character_name: str):
        """Load a character (either predefined or custom)."""
        name = character_name.lower().strip()
        
        # 实时重新加载配置，确保切换时能读取到最新配置
        global CHARACTERS_CONFIG
        CHARACTERS_CONFIG = load_characters_config()
        
        if name not in CHARACTERS_CONFIG:
            raise ValueError(
                f"未知角色: {name}。\n"
                f"可用角色列表请查看 characters.json 文件"
            )

        if name not in self._loaded_characters:
            logger.info(f"[TTS] Loading character: {name}...")
            config = CHARACTERS_CONFIG[name]

            if config.get("type", "predefined") == "custom":
                # 构建绝对路径，兼容在不同目录下运行脚本的情况
                # 如果 model_dir 是绝对路径也会正确识别，否则拼接为基于项目的绝对路径
                model_dir = config.get("model_dir")
                if not os.path.isabs(model_dir):
                    model_dir = os.path.join(BASE_DIR, model_dir)
                    
                ref_audio = config.get("ref_audio")
                if not os.path.isabs(ref_audio):
                    ref_audio = os.path.join(BASE_DIR, ref_audio)

                ref_text = config.get("ref_text")
                language = config.get("lang", "Chinese")
                
                self.load_custom_model(name, model_dir, ref_audio, ref_text, language=language)
            else:
                # 调用原生预设角色加载逻辑
                genie.load_predefined_character(name)
            
            self._loaded_characters.add(name)
            logger.info(f"[TTS] Character {name} loaded.")

        self._current_character = name

    def load_custom_model(self, name: str, model_dir: str,
                          ref_audio: str, ref_text: str, language: str = "zh"):
        """Load a custom ONNX model directory (Genie format)."""
        genie.load_character(
            character_name=name,
            onnx_model_dir=model_dir,
            language=language,
        )
        genie.set_reference_audio(
            character_name=name,
            audio_path=ref_audio,
            audio_text=ref_text,
            language=language,
        )
        self._loaded_characters.add(name)
        self._current_character = name

    @property
    def current_character(self) -> str:
        return self._current_character

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @staticmethod
    def clean_text(text: str) -> str:
        """清洗 EPUB 文本中会让 TTS 产生电音 / 空文件的字符。"""
        # Clean zero-width characters and spaces between Chinese characters
        # This prevents unnatural pauses in TTS caused by invisible formatting or layout spaces from EPUBs.
        text = re.sub(r'[\u200b\u200c\u200d\ufeff\xad]', '', text)
        text = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', text)
        
        # Clean up repeated punctuation (like multiple ellipsis or dots) which cause the TTS engine to hallucinate 
        # producing stuttering noises (恩恩。。嗯嗯嗯。。。), hoarseness, and electrical noise.
        text = re.sub(r'[…]+', '…', text)          # Multiple Chinese ellipsis to single
        text = re.sub(r'\.{2,}', '…', text)        # Multiple English dots to single ellipsis
        text = re.sub(r'([。！？\.\!\?])\1+', r'\1', text) # Repeated punctuations to single
        
        # 很多 TTS 推理在处理结尾为冒号（：）或分号（；）时容易切分异常导致生成空文件失败
        # 故将其全部替换为逗号，实现同样的语音停顿效果即可
        text = text.replace('：', '，').replace(':', ',')
        text = text.replace('；', '，').replace(';', ',')
        return text.strip()

    def synthesize(self, text: str, prev_text: str = "", next_text: str = "") -> bytes:
        """
        Synthesize text to WAV audio bytes.
        Returns: WAV file content as bytes.
        """
        if not text.strip():
            return b""

        text = self.clean_text(text)

        # 清洗之后可能只剩标点（例如原文就是 "……" 或 "——"）。
        # 这种文本送进 genie 会得到 0 字节文件，进而抛异常变成 500，
        # 在 Koodo 那边表现为一次莫名其妙的长时间卡顿。直接返回一小段静音。
        if not text or not SPEAKABLE.search(text):
            logger.info("[TTS] No speakable content after cleaning, returning silence.")
            return self._silent_wav()

        if not self._current_character:
            raise RuntimeError("No character loaded. Call load_character first.")

        # Check cache first (Hash to save memory in keys)
        cache_key = hashlib.md5(f"{self._current_character}_{text}_{prev_text}_{next_text}".encode()).hexdigest()
        with self._lock:
            if cache_key in self._cache:
                logger.info(f"[TTS] Cache hit for text: {text[:15]}...")
                self._cache_keys.remove(cache_key)
                self._cache_keys.append(cache_key)
                return self._cache[cache_key]

        # 关键：切分必须发生在这里，而不是只在 TextProcessor 里。
        # Koodo 插件是直接 POST /api/synthesize_text 的，送进来的是 Koodo 自己分好的
        # 文本块，完全不经过 TextProcessor。而 genie 内部的 TextSplitter 只在遇到标点
        # 时才切，碰到无标点长串会把整段丢给解码器 → 撞上 500 步上限 → 拦腰截断。
        segments = self._text_processor.split_for_tts(text)
        if not segments:
            return self._silent_wav()

        # genie 的 tts_player 是模块级单例，context.current_speaker 和内部队列都共享，
        # 所以整段合成期间必须一直持锁，否则两个并发请求的片段会互相串音。
        with self._lock:
            frames = bytearray()
            pause = b"\x00" * (int(self._sample_rate * INTER_SEGMENT_PAUSE_SEC) * 2)
            for segment in segments:
                segment_frames = self._synthesize_segment(segment)
                if not segment_frames:
                    logger.warning(f"[TTS] Segment produced no audio, skipped: {segment[:20]}...")
                    continue
                if frames:
                    frames += pause
                frames += segment_frames

        if not frames:
            logger.error(f"[TTS] All {len(segments)} segment(s) failed for: {text[:30]}...")
            return b""

        wav_bytes = self._build_wav(bytes(frames))

        # Save to cache
        with self._lock:
            if cache_key in self._cache:
                self._cache_keys.remove(cache_key)
            self._cache[cache_key] = wav_bytes
            self._cache_keys.append(cache_key)
            if len(self._cache_keys) > self._cache_max_size:
                oldest = self._cache_keys.pop(0)
                del self._cache[oldest]

        return wav_bytes

    def _synthesize_segment(self, segment: str, retries: int = 1) -> bytes:
        """合成单个片段，返回裸 PCM 帧。失败会重试一次。调用方需持有 self._lock。"""
        for attempt in range(retries + 1):
            # Use a temp file since Genie-TTS writes to file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                # 注意：genie.tts 在缺少参考音频时只写一条 error 日志就 return，并不抛异常；
                # 片段全部失败时它也不会调用 _save_session_audio，临时文件会停在 0 字节。
                # 所以这里必须自己校验产物，不能假定调用成功。
                genie.tts(
                    character_name=self._current_character,
                    text=segment,
                    play=False,
                    split_sentence=True,
                    save_path=tmp_path,
                )
                if os.path.getsize(tmp_path) > 44:  # 44 = WAV 头的大小
                    with wave.open(tmp_path, "rb") as wf:
                        self._sample_rate = wf.getframerate()
                        return wf.readframes(wf.getnframes())
                if attempt < retries:
                    logger.warning(f"[TTS] Empty output, retrying segment: {segment[:20]}...")
            except Exception as e:
                logger.error(f"[TTS] Segment synthesis failed ({e}): {segment[:20]}...")
                if attempt >= retries:
                    raise
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return b""

    def _build_wav(self, frames: bytes) -> bytes:
        """把裸 PCM 帧包成一个完整的 WAV 文件。"""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # genie 固定输出 16-bit 单声道
            wf.setframerate(self._sample_rate)
            wf.writeframes(frames)
        return buf.getvalue()

    def _silent_wav(self, duration: float = 0.25) -> bytes:
        """一小段静音，用来替代本该报错的空音频。"""
        return self._build_wav(b"\x00" * (int(self._sample_rate * duration) * 2))

    def get_characters(self) -> dict:
        """Return available characters."""
        config = load_characters_config()
        return {
            k: {**v, "loaded": k in self._loaded_characters}
            for k, v in config.items()
        }
