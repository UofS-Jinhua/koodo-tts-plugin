"""
TTS Engine wrapper for Genie-TTS (GPT-SoVITS ONNX Inference).
Provides a clean interface for text-to-speech synthesis.
"""
import os
import io
import wave
import logging
import tempfile
import soundfile as sf
import numpy as np
import threading
import hashlib
import genie_tts as genie
from text_processor import TextProcessor

from emotion_analyzer import analyzer
import json

logger = logging.getLogger(__name__)

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

class TTSEngine:
    """Wraps Genie-TTS for GPT-SoVITS inference."""

    def __init__(self, character: str = "kiana"):
        self._loaded_characters = set()
        self._current_character = None
        self._sample_rate = 32000  # GPT-SoVITS default
        self._text_processor = TextProcessor()
        self._lock = threading.Lock()
        self._emotion_lock = threading.Lock()
        
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

    def synthesize(self, text: str, prev_text: str = "", next_text: str = "") -> bytes:
        """
        Synthesize text to WAV audio bytes.
        Returns: WAV file content as bytes.
        """
        if not text.strip():
            return b""

        import re
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

        # ---- Dynamic Emotion Audio Switch ----
        # 1. Analyze the emotion of the current sentence with context (Using separate lock to prevent CPU thrashing)
        with self._emotion_lock:
            emotion = analyzer.analyze(text, prev_text, next_text)
        
        # 2. Compose the target character key based on the detected emotion
        # For example, if current_character is "kiana" and emotion is "happy", the key is "kiana_happy".
        # If emotion is "neutral", fallback to "kiana"
        target_char_key = self._current_character if emotion == "neutral" else f"{self._current_character}_{emotion}"
        
        global CHARACTERS_CONFIG
        # 3. Get configuration for target emotion (fallback to base character if not found)
        config = CHARACTERS_CONFIG.get(target_char_key, CHARACTERS_CONFIG.get(self._current_character))
        
        # 4. If the fallback config has custom reference audio, dynamically apply it
        if config and config.get("type", "predefined") == "custom":
            ref_audio = config.get("ref_audio")
            ref_text = config.get("ref_text")
            lang = config.get("lang", "Chinese")
            
            if ref_audio and not os.path.isabs(ref_audio):
                ref_audio = os.path.join(BASE_DIR, ref_audio)
                
            if ref_audio and ref_text:
                # We replace the characteristics without reloading the ONNX models!
                with self._lock:
                    genie.set_reference_audio(
                        character_name=self._current_character,
                        audio_path=ref_audio,
                        audio_text=ref_text,
                        language=lang,
                    )

        # Use a temp file since Genie-TTS writes to file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with self._lock:
                genie.tts(
                    character_name=self._current_character,
                    text=text,
                    play=False,
                    split_sentence=True,
                    save_path=tmp_path,
                )

            # Read back and return as bytes
            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()

            # Get actual sample rate from the WAV file
            data, sr = sf.read(tmp_path)
            self._sample_rate = sr
            
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

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def get_characters(self) -> dict:
        """Return available characters."""
        config = load_characters_config()
        return {
            k: {**v, "loaded": k in self._loaded_characters}
            for k, v in config.items()
        }
