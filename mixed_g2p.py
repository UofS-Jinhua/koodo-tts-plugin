"""
让 genie-tts 支持中英混读。

genie-tts 2.0.2 的语言是「按角色」锁死的：genie.tts() 没有 language 参数，
合成时用的是加载角色时传入的那一个语种，然后在
genie_tts/GetPhonesAndBert.py 里用一个 if/elif/else 把**整段文本**丢给
中文 / 英文 / 日文三选一的 G2P。而中文 G2P 会主动删掉所有英文：

  * ChineseG2P.py 的 pattern_filter 只保留 [一-龥] 和几个标点；
  * g2p() 里还有一句 seg = self.pattern_eng.sub("", seg)（注释写着「移除英文」）；
  * jieba 词性为 eng 的词直接 continue。

所以「中文句子里夹的英文被整段跳过」是上游的设计，不是模型的能力问题——
GPT-SoVITS v2 的符号表 (SymbolsV2.py) 里拼音、罗马音、ARPAbet 是同一张表，
模型本身完全会发英文音，缺的只是把中文段落里的英文片段路由到英文 G2P。

本模块就补上这一步：把 get_phones_and_bert 换成按语种分段的版本，
中文片段走 ChineseG2P + RoBERTa 特征，英文片段走 EnglishG2P + 零特征
（和上游 GPT-SoVITS 的 zh/en 混合模式一致，英文没有中文 BERT 特征可用），
最后把音素序列和 BERT 特征沿时间轴拼起来，仍然只跑一次推理。

用法：在 import genie_tts 之后调用一次 apply()。这是运行时猴补丁，
不改 site-packages 里的源码，重建虚拟环境后依然生效。
"""
import logging
import re
from typing import Callable, List, Optional, Tuple

import numpy as np

import genie_tts.Audio.ReferenceAudio as _reference_audio_module
import genie_tts.Core.Inference as _inference_module
import genie_tts.GetPhonesAndBert as _gpb_module
from genie_tts.ModelManager import model_manager
from genie_tts.Utils.Constants import BERT_FEATURE_DIM
from genie_tts.Utils.Language import normalize_language

logger = logging.getLogger(__name__)

# 两个调用点都是 `from ..GetPhonesAndBert import get_phones_and_bert`，
# 名字已经绑定到各自的模块命名空间里了，所以三个模块都得打上补丁。
_PATCH_TARGETS = (_gpb_module, _inference_module, _reference_audio_module)
_ORIGINAL_GET_PHONES_AND_BERT = _gpb_module.get_phones_and_bert

# 一个「英文词」：必须以字母开头，允许词内的 ' - . 连接后续字母/数字，
# 这样 don't / Wi-Fi / GPT-4 / U.S.A 会被当成一个整体（英文 G2P 读起来更自然），
# 而句末的 "it." 里的句号不会被吞掉（. 后面必须还有字母或数字）。
_EN_WORD = r"[A-Za-z]+(?:['’\-.][A-Za-z0-9]+)*"
# 仅由空格分隔的连续英文词合并成一段，让 pos_tag / 连读判断能看到上下文
# （"machine learning" 比 "machine" + "learning" 两次调用读得更连贯）。
_EN_RUN = re.compile(r"%s(?:[ \t]+%s)*" % (_EN_WORD, _EN_WORD))

# 英文 G2P 惰性加载：它要读 GenieData/G2P/EnglishG2P 下的词典和 nltk 词性标注器，
# 加载一次要几秒，纯中文的书就没必要付这个代价。
_english_to_phones: Optional[Callable[[str], List[int]]] = None
_english_unavailable = False


def split_language_runs(text: str) -> List[Tuple[str, str]]:
    """把文本切成 ('zh'|'en', 片段) 的序列。英文以外的一切都算中文片段。

    数字保持在中文片段里（"2024年" 读「二〇二四年」而不是 "twenty twenty-four"），
    只有紧贴着英文词的数字才会跟着走英文（"GPT-4" → "gee pee tee four"）。
    """
    runs: List[Tuple[str, str]] = []
    pos = 0
    for match in _EN_RUN.finditer(text):
        if match.start() > pos:
            runs.append(("zh", text[pos:match.start()]))
        runs.append(("en", match.group()))
        pos = match.end()
    if pos < len(text):
        runs.append(("zh", text[pos:]))
    return runs


def _zero_bert(rows: int) -> np.ndarray:
    return np.zeros((rows, BERT_FEATURE_DIM), dtype=np.float32)


def _fit_rows(bert: np.ndarray, expected: int) -> np.ndarray:
    """保证 BERT 特征行数与音素数严格一致。

    ChineseG2P.process() 末尾会用 `ph in symbols_v2` 再过滤一遍音素，
    理论上可能让 len(phones) 和 sum(word2ph) 对不上。原实现里对不上顶多是
    这一句报错，而这里是要把多段拼起来的，一段错位会毁掉整句，所以兜一下底。
    """
    rows = bert.shape[0]
    if rows == expected:
        return bert
    logger.warning("[mixed-g2p] BERT 行数 %d 与音素数 %d 不符，已对齐。", rows, expected)
    if rows > expected:
        return bert[:expected]
    return np.concatenate([bert, _zero_bert(expected - rows)], axis=0)


def _chinese_run(text: str) -> Tuple[List[int], np.ndarray]:
    """中文片段：音素 + RoBERTa 特征。等价于原 get_phones_and_bert 的中文分支。"""
    from genie_tts.G2P.Chinese.ChineseG2P import chinese_to_phones

    text_clean, _, phones, word2ph = chinese_to_phones(text)
    if not phones:
        return [], _zero_bert(0)

    if model_manager.load_roberta_model():
        try:
            encoded = model_manager.roberta_tokenizer.encode(text_clean)
            outputs = model_manager.roberta_model.run(None, {
                'input_ids': np.array([encoded.ids], dtype=np.int64),
                'attention_mask': np.array([encoded.attention_mask], dtype=np.int64),
                'repeats': np.array(word2ph, dtype=np.int64),
            })
            return phones, _fit_rows(outputs[0].astype(np.float32), len(phones))
        except Exception as exc:  # RoBERTa 只是锦上添花，挂了也不该让整句合成失败
            logger.warning("[mixed-g2p] RoBERTa 推理失败，该片段退回零特征: %s", exc)

    return phones, _zero_bert(len(phones))


def _english_run(text: str) -> Tuple[List[int], np.ndarray]:
    """英文片段：ARPAbet 音素 + 零 BERT 特征（中文 RoBERTa 对英文没有意义）。"""
    phones = _english_to_phones(text)
    return phones, _zero_bert(len(phones))


def _load_english_g2p() -> Optional[Callable[[str], List[int]]]:
    global _english_to_phones, _english_unavailable
    if _english_to_phones is None and not _english_unavailable:
        try:
            from genie_tts.G2P.English.EnglishG2P import english_to_phones
        except Exception as exc:
            # 缺 GenieData/G2P/EnglishG2P 之类的资源问题：记一次日志就彻底放弃，
            # 之后每句话都退回原实现（英文照旧被跳过），至少中文还能正常读。
            logger.warning("[mixed-g2p] 英文 G2P 不可用，中英混读已禁用: %s", exc)
            _english_unavailable = True
        else:
            _english_to_phones = english_to_phones
    return _english_to_phones


def get_phones_and_bert(prompt_text: str, language: str = 'japanese') -> Tuple[np.ndarray, np.ndarray]:
    """按语种分段的 get_phones_and_bert。只接管中文角色，其余原样转交上游。"""
    if normalize_language(language) != 'Chinese':
        return _ORIGINAL_GET_PHONES_AND_BERT(prompt_text, language=language)

    runs = split_language_runs(prompt_text)
    if not any(kind == "en" for kind, _ in runs):
        return _ORIGINAL_GET_PHONES_AND_BERT(prompt_text, language=language)
    if _load_english_g2p() is None:
        return _ORIGINAL_GET_PHONES_AND_BERT(prompt_text, language=language)

    all_phones: List[int] = []
    all_bert: List[np.ndarray] = []
    for kind, segment in runs:
        if not segment.strip():
            continue
        try:
            phones, bert = _english_run(segment) if kind == "en" else _chinese_run(segment)
        except Exception as exc:
            logger.warning("[mixed-g2p] %s 片段 G2P 失败，已跳过 (%r): %s", kind, segment[:20], exc)
            continue
        if not phones:
            continue
        all_phones.extend(phones)
        all_bert.append(bert)

    if not all_phones:
        return _ORIGINAL_GET_PHONES_AND_BERT(prompt_text, language=language)

    return np.array([all_phones], dtype=np.int64), np.concatenate(all_bert, axis=0)


get_phones_and_bert._genie_mixed_g2p = True  # type: ignore[attr-defined]


def is_applied() -> bool:
    return getattr(_gpb_module.get_phones_and_bert, "_genie_mixed_g2p", False)


def apply() -> bool:
    """打上补丁。返回 True 表示这次真的打了，False 表示之前已经打过。"""
    if is_applied():
        return False
    for module in _PATCH_TARGETS:
        module.get_phones_and_bert = get_phones_and_bert
    logger.info("[mixed-g2p] 已启用中英混读。")
    return True


def revert() -> None:
    """还原成 genie 原本的行为（英文会被丢掉），主要给排查问题用。"""
    for module in _PATCH_TARGETS:
        module.get_phones_and_bert = _ORIGINAL_GET_PHONES_AND_BERT
