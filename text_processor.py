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

    def __init__(self):
        self._documents: Dict[str, dict] = {}

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

    def _split_sentences(self, text: str) -> List[str]:
        parts = self.SENTENCE_ENDINGS.split(text)
        sentences, current = [], ""
        for part in parts:
            if not part:
                continue
            if self.SENTENCE_ENDINGS.fullmatch(part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                if current.strip():
                    sentences.append(current.strip())
                current = part
        if current.strip():
            if len(current.strip()) > 80:
                sentences.extend(self._split_long(current.strip()))
            else:
                sentences.append(current.strip())
        return sentences

    def _split_long(self, text: str, max_len: int = 120) -> List[str]:
        if len(text) <= max_len:
            return [text]
        parts = re.split(r'([，,、]+)', text)
        sentences, current = [], ""
        for part in parts:
            if len(current) + len(part) <= max_len:
                current += part
            else:
                if current.strip():
                    sentences.append(current.strip())
                current = part
        if current.strip():
            sentences.append(current.strip())
        return sentences if sentences else [text]

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
