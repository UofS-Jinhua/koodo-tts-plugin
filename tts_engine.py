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
import ort_memory_patch
from text_processor import TextProcessor

import json

logger = logging.getLogger(__name__)

# genie-tts 的中文 G2P 会把句子里夹带的英文整段删掉（详见 mixed_g2p 的模块说明），
# 必须在加载角色（会给参考文本做 G2P）之前把补丁打上。
mixed_g2p.apply()

# genie-tts 在内存里把 fp16 权重转成 fp32 再从字节建 session，常驻约 3 倍权重，
# 而且 8 个 session 各有一个只涨不缩的内存池——只加载 kiana 就占 8.6 GB。改成
# 加载一次性转好的 fp32 缓存（ModelCache/fp32/）、所有 session 共用一个内存池
# 后是 4.2 GB，合成速度不变（详见 ort_memory_patch 的模块说明）。session 在加载
# 角色时创建，所以同样必须在加载任何东西之前打上补丁。
ort_memory_patch.apply()

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
# 判定"解码器早停"的时长下限，单位：秒/字。详见 _min_plausible_sec。
# 实测正常合成的下限是 0.164 秒/字（8 个代表性片段 × 3 次，split 两种模式都测过），
# 这里取它的一半，既能抓住塌陷又不会误判正常的快句子。
MIN_SEC_PER_CHAR = 0.082
# 至少要有一个能发音的字符，否则送进 TTS 只会得到 0 字节文件
# (\w 在 Python 3 的 str 模式下已经涵盖汉字、假名等 Unicode 字母)
SPEAKABLE = re.compile(r'\w')

# genie 的 TextSplitter 只认 “”‘’"' 这几种成对符号 (见 genie_tts/Utils/
# TextSplitter.py 的 all_puncts_chars)，中文直角引号和各类括号一概不认。后果是
# 它会把 「你走吧。」 切成 「你走吧。 和 」 两句，而 get_effective_len 又把落单的
# 」 当成宽度 2 的"内容"字符，于是这个纯标点片段被当作正经句子送去合成：G2P 把
# 它规范化成空，解码器没有任何文本可依据，就会凭空生成 7 秒左右的气声。
# 实测 」 单独合成 = 7.28 秒垃圾音频；「你走吧。」 整句时长在 0.8s 和 8.1s 之间
# 双峰跳变。而且 genie 的 worker 循环 catch 住单句异常后只记日志就跳到下一句
# (Core/TTSPlayer.py)，丢掉的那句不会让整体失败——表现就是"有时候吞字"。
# 同一类问题在 （…） 上也复现过（尾部落单的 ）），所以括号family 一并处理。
# 这些符号本身都不发音，引号携带的旁白/台词信息又已经在 TextProcessor 切分时
# 用掉了，送进 TTS 之前一律去掉。
UNSPOKEN_MARKS = str.maketrans("", "", "「」『』（）()【】〔〕〖〗《》〈〉[]{}")

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

        # 破折号必须自己降级成逗号，genie 那边指望不上。
        # ChineseG2P 的 pattern_filter 只保留 ! ? … , . -，其余非汉字字符整段删掉
        # (见 G2P/Chinese/ChineseG2P.py 的 PUNCTUATION / pattern_filter)。它的
        # PUNCTUATION_REPLACEMENTS 里虽然写了 "—": "-"，但破折号在映射之前就被
        # TextNormalizer 吃掉了，那条规则是死代码——实测
        #   "你说这个啊——普通血族" -> text_clean "你说这个啊普通血族"
        # 破折号连同它该有的停顿一起消失，两个分句直接黏在一起念。
        # 这里降级成逗号，把停顿还回来。
        text = re.sub(r'[—－–]{1,}|_{2,}', '，', text)
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

        # 注意：这里不对整个 segments 列表持锁——每个 genie.tts() 调用自己在
        # _synthesize_segment 内部加锁（见那里的说明）。Koodo 会并发预取好几句，
        # 如果在这里把整段循环锁住，一句长段落（尤其是现在引号切分之后，一段
        # 可能拆成十几个片段）就会独占锁几十秒，把其他本该几乎秒回的短句请求
        # 全部堵住——实测发生过：一次请求连续合成 13 个片段耗时 61 秒，同一时间
        # 排队的 8 个请求全部卡到那 61 秒结束才一起返回。Koodo 等不到它正要播的
        # 那一句，播放就断在那里，而服务端这边看到的只是"全部最终都 200 了"。
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

    def _min_plausible_sec(self, segment: str) -> float:
        """这段文本至少该念多久。低于这个时长就说明解码器早停、内容被吞了。

        实测正常朗读速度约 0.20~0.37 秒/字（短片段因为有起音和收尾偏慢），
        这里按 0.06 判，留了 3 倍以上余量：正常合成不会误判，而早停的产物
        （23 个字只出 0.20 秒 ≈ 0.009 秒/字）会被稳稳抓住。
        """
        return TextProcessor.width(segment) / 2 * MIN_SEC_PER_CHAR

    def _synthesize_segment(self, segment: str, retries: int = 2) -> bytes:
        """合成单个片段，返回裸 PCM 帧。失败会重试。"""
        # 见 UNSPOKEN_MARKS 的说明。必须在这里去掉而不是在 clean_text 里：
        # clean_text 跑在切分之前，TextProcessor 还要靠引号区分旁白和台词。
        segment = segment.translate(UNSPOKEN_MARKS).strip()
        if not segment or not SPEAKABLE.search(segment):
            return b""

        min_sec = self._min_plausible_sec(segment)
        best = b""  # 全部尝试都不合格时，至少把最长的那次交出去，不要退化成整段静音
        for attempt in range(retries + 1):
            # Use a temp file since Genie-TTS writes to file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                # 注意：genie.tts 在缺少参考音频时只写一条 error 日志就 return，并不抛异常；
                # 片段全部失败时它也不会调用 _save_session_audio，临时文件会停在 0 字节。
                # 所以这里必须自己校验产物，不能假定调用成功。
                #
                # 锁只包住这一次调用，不包住整个重试循环、更不包住调用方的整个片段列表：
                # genie 的 tts_player 是模块级单例（context.current_speaker、内部队列都
                # 共享），两次 genie.tts() 调用不能同时进行，否则会互相踩状态、串音——
                # 但「不能同时」指的是调用本身，不要求同一个请求的所有片段之间不被别的
                # 请求插队。缩小到这里之后，一句长段落不会再占着锁让其他并发请求干等；
                # 每次调用之间会公平地把锁让给排队的其他线程。
                # split_sentence=False：不让 genie 再切一刀，一次调用 = 一次生成。
                # 这不是为了省时间（实测两种模式 rtf 0.772 vs 0.797，在噪声里），
                # 而是为了让下面的时长校验真正兜得住：genie 内部会把约 11% 的片段
                # 再切成 2 块，各自独立生成，其中一块塌掉时整段只短了一半左右，
                # 按整段算的时长校验根本发现不了，那半句就被静默吞掉了。
                # 关掉之后塌陷只会是整块的，特征极明显（实测塌陷约 0.011 秒/字，
                # 而正常下限是 0.164），一抓一个准。
                # 切分的活我们自己在 TextProcessor 里干，MAX_SEGMENT_WIDTH=70
                # 约合 35 个汉字 ≈ 175 解码步，离 500 步上限还很远。
                with self._lock:
                    genie.tts(
                        character_name=self._current_character,
                        text=segment,
                        play=False,
                        split_sentence=False,
                        save_path=tmp_path,
                    )
                frames = b""
                if os.path.getsize(tmp_path) > 44:  # 44 = WAV 头的大小
                    with wave.open(tmp_path, "rb") as wf:
                        self._sample_rate = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())

                # GPT-SoVITS 的自回归解码器偶发早停：EOS 提前触发，几百毫秒就收尾，
                # 整句内容被吞掉。这在 genie 那边不算失败（产物是合法 WAV，只是很短），
                # 光看文件大小根本发现不了，必须按文本长度校验时长。
                if len(frames) / 2 / self._sample_rate >= min_sec:
                    return frames
                if len(frames) > len(best):
                    best = frames
                if attempt < retries:
                    reason = "Empty output" if not frames else "Suspiciously short output (decoder stopped early)"
                    logger.warning(f"[TTS] {reason}, retrying segment: {segment[:20]}...")
            except Exception as e:
                logger.error(f"[TTS] Segment synthesis failed ({e}): {segment[:20]}...")
                if attempt >= retries:
                    raise
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        if best:
            logger.error(f"[TTS] Segment still too short after {retries + 1} attempts, "
                         f"audio may be truncated: {segment[:30]}...")
        return best

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
