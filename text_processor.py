"""
Text processing utilities for the audiobook app.
Handles text file parsing and intelligent sentence splitting.
"""
import re
import uuid
from typing import List, Dict, Optional


class TextProcessor:
    """Processes text files into sentences for TTS."""

    SENTENCE_ENDINGS = re.compile(r'([。！？…；\.\!\?\;]+["\'\）\)」』】]?)\s*')
    PARAGRAPH_SEP = re.compile(r'\n\s*\n')
    # 次级分隔符：句子仍然超长时在这里切
    CLAUSE_SEP = re.compile(r'([，,、；;：:—]+)')

    # GPT-SoVITS 的自回归解码器在 genie-tts 里被写死为最多 500 步
    # (见 genie_tts/Core/Inference.py 的 `for idx in range(0, 500)`)，
    # 跑满之后不报错、直接返回半截音频。500 步约等于 10 秒 / 45~50 个汉字。
    # 这里按“显示宽度”限制（汉字算 2、ASCII 算 1），留足安全余量。
    MAX_SEGMENT_WIDTH = 70

    def __init__(self):
        self._documents: Dict[str, dict] = {}

    @staticmethod
    def width(text: str) -> int:
        """显示宽度：非 ASCII（中日韩）算 2，ASCII 算 1。"""
        return sum(2 if ord(c) > 127 else 1 for c in text)

    def load_text(self, text: str, filename: str = "untitled.txt") -> str:
        doc_id = str(uuid.uuid4())[:8]
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Clean zero-width chars and spacing between Chinese characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff\xad]', '', text)
        text = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', text)
        paragraphs = self.PARAGRAPH_SEP.split(text)
        sentences = []
        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            for sent in self._split_sentences(para):
                sent = sent.strip()
                if sent:
                    sentences.append({"id": len(sentences), "text": sent, "paragraph": para_idx})
        self._documents[doc_id] = {"id": doc_id, "filename": filename, "sentences": sentences, "total_sentences": len(sentences)}
        return doc_id

    def split_for_tts(self, text: str) -> List[str]:
        """把任意文本切成保证不会触发解码器截断的片段。"""
        segments = []
        for para in self.PARAGRAPH_SEP.split(text):
            para = para.strip()
            if para:
                segments.extend(self._split_sentences(para))
        return segments

    def _split_sentences(self, text: str) -> List[str]:
        parts = self.SENTENCE_ENDINGS.split(text)
        sentences, current = [], ""

        def flush():
            # 每一个切出来的句子都要过长度检查，而不只是最后的残余部分
            if current.strip():
                sentences.extend(self._split_long(current.strip()))

        for part in parts:
            if not part:
                continue
            if self.SENTENCE_ENDINGS.fullmatch(part):
                current += part
                flush()
                current = ""
            else:
                flush()
                current = part
        flush()
        return sentences

    def _split_long(self, text: str, max_width: int = None) -> List[str]:
        max_width = max_width or self.MAX_SEGMENT_WIDTH
        if self.width(text) <= max_width:
            return [text]

        # 第一轮：在逗号/顿号等次级标点处切
        parts, chunks, current = self.CLAUSE_SEP.split(text), [], ""
        for part in parts:
            if not part:
                continue
            if current and self.width(current) + self.width(part) > max_width:
                chunks.append(current)
                current = part
            else:
                current += part
        if current:
            chunks.append(current)

        # 第二轮：仍然超长的（无标点长串）按宽度硬切
        result = []
        for chunk in chunks or [text]:
            chunk = chunk.strip()
            if not chunk:
                continue
            if self.width(chunk) <= max_width:
                result.append(chunk)
            else:
                result.extend(self._hard_split(chunk, max_width))
        return result or [text]

    @classmethod
    def _hard_split(cls, text: str, max_width: int) -> List[str]:
        """无标点可切时的兜底：按显示宽度均匀硬切。"""
        out, current, current_width = [], "", 0
        for char in text:
            char_width = cls.width(char)
            if current and current_width + char_width > max_width:
                out.append(current)
                current, current_width = "", 0
            current += char
            current_width += char_width
        if current:
            out.append(current)
        return out

    def get_document(self, doc_id: str) -> Optional[dict]:
        return self._documents.get(doc_id)

    def get_sentence(self, doc_id: str, sentence_id: int) -> Optional[str]:
        doc = self._documents.get(doc_id)
        if doc and 0 <= sentence_id < len(doc["sentences"]):
            return doc["sentences"][sentence_id]["text"]
        return None

    def get_sentences(self, doc_id: str) -> Optional[List[dict]]:
        doc = self._documents.get(doc_id)
        return doc["sentences"] if doc else None
