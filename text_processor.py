"""
Text processing utilities for the audiobook app.
Handles text file parsing and intelligent sentence splitting.
"""
import re
import uuid
from typing import List, Dict, Optional, Tuple


class TextProcessor:
    """Processes text files into sentences for TTS."""

    # 结尾标点后面可以跟一个闭引号/右括号，一起算作句子的一部分，
    # 否则 “别过来！”她后退了一步。 会把 ” 甩到下一句开头。
    SENTENCE_ENDINGS = re.compile(r'([。！？…；\.\!\?\;]+["\'”’\）\)」』】]?)\s*')
    PARAGRAPH_SEP = re.compile(r'\n\s*\n')
    # 次级分隔符：句子仍然超长时在这里切
    CLAUSE_SEP = re.compile(r'([，,、；;：:—]+)')

    # 台词引号：只认成对的中文引号。英文直引号和撇号在中文小说里分不开，不参与切分。
    QUOTE_PAIRS = {'「': '」', '『': '』', '“': '”'}

    # 片段类型。台词和旁白将来可以配不同的参考音频 / 停顿长度。
    DIALOGUE = "dialogue"
    NARRATION = "narration"

    # 引号切分后，比这个显示宽度还短的碎片不单独成段（例如“他：”），并回相邻片段。
    # 太短的片段送进 GPT-SoVITS 既不稳，多出来的停顿听着也别扭。
    # 这个阈值得靠耳朵调：调大 → 碎片更少但混合段更多；调小 → 旁白/台词分得更干净。
    MIN_QUOTE_SEGMENT_WIDTH = 6

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
            for sent, style in self._split_sentences(para):
                sent = sent.strip()
                if sent:
                    sentences.append({
                        "id": len(sentences),
                        "text": sent,
                        "style": style,
                        "paragraph": para_idx,
                    })
        self._documents[doc_id] = {"id": doc_id, "filename": filename, "sentences": sentences, "total_sentences": len(sentences)}
        return doc_id

    def split_for_tts(self, text: str) -> List[str]:
        """把任意文本切成保证不会触发解码器截断的片段。

        TTS 合成的热路径：不做引号切分（见 split_quotes=False 的说明），只按
        句末标点切，然后把相邻的短片段打包（见 _pack_segments）——碎片段是
        rtf 杀手，必须合并掉。dialogue/narration 这个标记目前没有任何地方在用
        （没有接旁白/台词切换语音），纯粹是白付出的开销，所以这条热路径上
        关掉引号切分。那套逻辑还留着给 split_for_tts_with_style 用——真的要做
        旁白/台词切换语音的时候，把这里换回 True 即可。
        """
        segments = []
        for para in self.PARAGRAPH_SEP.split(text):
            para = para.strip()
            if not para:
                continue
            pieces = [seg for seg, _ in self._split_sentences(para, split_quotes=False)]
            segments.extend(self._pack_segments(pieces))
        return segments

    @classmethod
    def _pack_segments(cls, pieces: List[str], max_width: int = None) -> List[str]:
        """把相邻片段贪心合并到接近 MAX_SEGMENT_WIDTH，减少每块的固定开销。

        每次 genie.tts() 调用都要重跑一遍 encoder + first-stage decoder，实测
        约 0.218 秒的固定开销，跟文本长短无关。拟合出来的关系是
            合成耗时 ≈ 0.746 × 音频时长 + 0.218
        换算成 rtf（合成耗时 / 音频时长，>1 就意味着合成追不上朗读）：
            rtf ≈ 0.746 + 1.147 / 字数
        所以 3 个字的碎片 rtf ≈ 1.13，而 35 个字的整块 rtf ≈ 0.78。

        这不是理论推演——`“哦……吸溜……你做了个怪梦？吸溜……”` 这种全是省略号的
        对话，因为 SENTENCE_ENDINGS 把 … 当句末标点，会被切成一堆 3 个字的碎片，
        实测整段 rtf = 1.084（26.06s 合成 / 24.04s 音频）。合成比朗读还慢，
        Koodo 的缓冲必然被抽干，然后播放就停在那里。

        合并不会丢停顿：逗号、句号、省略号都还在文本里，genie 的 ChineseG2P
        照样把它们映射成停顿，只是不再为每 3 个字付一次固定开销。而且块数变少
        之后，撞上"偶发塌陷"（约 1%/块）的机会也跟着变少。
        """
        max_width = max_width or cls.MAX_SEGMENT_WIDTH
        packed: List[str] = []
        for piece in pieces:
            if packed and cls.width(packed[-1]) + cls.width(piece) <= max_width:
                packed[-1] += piece
            else:
                packed.append(piece)
        return packed

    def split_for_tts_with_style(self, text: str, split_quotes: bool = True) -> List[Tuple[str, str]]:
        """同 split_for_tts，但每个片段附带 DIALOGUE / NARRATION 标记。"""
        segments = []
        for para in self.PARAGRAPH_SEP.split(text):
            para = para.strip()
            if para:
                segments.extend(self._split_sentences(para, split_quotes))
        return segments

    def _split_sentences(self, text: str, split_quotes: bool = True) -> List[Tuple[str, str]]:
        parts = self.SENTENCE_ENDINGS.split(text)
        sentences, current = [], ""

        def flush():
            # 每一个切出来的句子都要过长度检查，而不只是最后的残余部分
            if not current.strip():
                return
            piece_list = self._split_quotes(current.strip()) if split_quotes else [current.strip()]
            for piece in piece_list:
                piece = piece.strip()
                if not piece:
                    continue
                style = self.DIALOGUE if piece[0] in self.QUOTE_PAIRS else self.NARRATION
                # 超长台词被拆成多块之后，每一块都继承同一个 style，
                # 否则第二块开头没有引号，会被误判成旁白。
                sentences.extend((chunk, style) for chunk in self._split_long(piece))

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

    @classmethod
    def _split_quotes(cls, text: str) -> List[str]:
        """按引号边界把一句话切成「纯台词」和「纯旁白」两类片段。

        句号切分已经能分开 `「你走吧。」他转身离开了。` 这种；这里补的是句号切
        不开的混合句：`他说：「你走吧。」` 和 `「你走吧，」他说。`

        只在最外层引号处切（嵌套引号不切），拼回去和原文完全一致、不丢字符。
        引号一直没闭合时（跨段落的长台词），从引号处到结尾整体算作台词。
        """
        pieces, current, depth, expect_close = [], "", 0, None
        for char in text:
            if depth == 0:
                if char in cls.QUOTE_PAIRS:
                    # 进入台词：先把前面攒的旁白收掉
                    if current:
                        pieces.append(current)
                    current, depth, expect_close = char, 1, cls.QUOTE_PAIRS[char]
                else:
                    current += char
                continue

            current += char
            if char == expect_close:
                depth -= 1
                if depth == 0:
                    pieces.append(current)
                    current, expect_close = "", None
            elif cls.QUOTE_PAIRS.get(char) == expect_close:
                # 同种引号套同种引号（少见），跟着计数以免提前闭合
                depth += 1
        if current:
            pieces.append(current)
        return cls._merge_short(pieces)

    @classmethod
    def _merge_short(cls, pieces: List[str]) -> List[str]:
        """把过短的碎片并回相邻片段，避免“他：”这种两字片段单独成段。"""
        out: List[str] = []
        for piece in pieces:
            # 自己太短 → 并进前一片；前一片太短 → 自己并过去（并掉的是开头的引导语）
            if out and (cls.width(piece.strip()) < cls.MIN_QUOTE_SEGMENT_WIDTH
                        or cls.width(out[-1].strip()) < cls.MIN_QUOTE_SEGMENT_WIDTH):
                out[-1] += piece
            else:
                out.append(piece)
        return out

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
