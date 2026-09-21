# -*- coding: utf-8 -*-
"""
引号感知切分的自检脚本（纯文本处理，不加载任何模型，秒出结果）。

    .venv-genie\\Scripts\\python.exe tests\\test_split.py

检查两件事：
  1. 每个片段的 旁白 / 台词 标记对不对；
  2. 切完之后不丢字符、不超长（超长会被 GPT-SoVITS 解码器截断成半截音频）。
"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from text_processor import TextProcessor

D = TextProcessor.DIALOGUE
N = TextProcessor.NARRATION

LONG_LINE = "你要知道，这条路我走了整整十年，" * 3 + "从来没有一天想过要回头。"

CASES = [
    # 句号切分本来就能分开的（回归用，别把原有行为改坏了）
    ("「你走吧。」他转身离开了。",
     [("「你走吧。」", D), ("他转身离开了。", N)]),

    # 冒号引导：句号切不开，靠引号切
    ("他说：「你走吧。」",
     [("他说：", N), ("「你走吧。」", D)]),

    # 引号内是逗号：句号也切不开
    ("「你走吧，」他说。",
     [("「你走吧，」", D), ("他说。", N)]),

    # 全角双引号同样处理
    ("“别过来！”她后退了一步。",
     [("“别过来！”", D), ("她后退了一步。", N)]),

    # 旁白夹在两句台词中间
    ("「我知道。」他顿了顿，「但我不想说。」",
     [("「我知道。」", D), ("他顿了顿，", N), ("「但我不想说。」", D)]),

    # 嵌套引号：只在最外层切，内层原样保留
    ("他说：「她喊了句『快跑』就不见了。」",
     [("他说：", N), ("「她喊了句『快跑』就不见了。」", D)]),

    # 引导语太短（宽度 < MIN_QUOTE_SEGMENT_WIDTH）→ 并回台词，整段按旁白算
    ("他：「走。」",
     [("他：「走。」", N)]),

    # 引号没闭合（跨段落的长台词）：不丢字符，整段算台词
    ("「这段话很长，一直说到段落结束都没有收尾",
     [("「这段话很长，一直说到段落结束都没有收尾", D)]),

    # 纯旁白不受影响
    ("他走到窗边，推开了玻璃窗。",
     [("他走到窗边，推开了玻璃窗。", N)]),
]

PASSAGE = """他站在门口，犹豫了很久。

「你真的想清楚了吗？」老人问。

他说：「想清楚了。」然后推开门走了出去。风很大，吹得他睁不开眼。

「等等！」老人在身后喊，「你的伞！」
"""


def check_cases():
    tp = TextProcessor()
    failures = 0
    for text, expected in CASES:
        actual = tp.split_for_tts_with_style(text)
        ok = actual == expected
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
        if not ok:
            print(f"      期望: {expected}")
            print(f"      实际: {actual}")
    return failures


def check_invariants():
    """不丢字符 + 不超长。这两条错了会直接听出来（漏字、半句戛然而止）。"""
    tp = TextProcessor()
    failures = 0

    segments = tp.split_for_tts_with_style(PASSAGE)
    strip_ws = lambda s: re.sub(r"\s+", "", s)
    joined = strip_ws("".join(seg for seg, _ in segments))
    if joined != strip_ws(PASSAGE):
        failures += 1
        print("FAIL  切分丢字符")
        print(f"      原文: {strip_ws(PASSAGE)}")
        print(f"      切完: {joined}")
    else:
        print("PASS  切分不丢字符")

    # 超长台词被拆成多块之后，每一块都得还是台词
    long_quote = f"他说：「{LONG_LINE}」"
    chunks = tp.split_for_tts_with_style(long_quote)
    dialogue_chunks = [seg for seg, style in chunks if style == D]
    if len(chunks) < 3 or len(dialogue_chunks) != len(chunks) - 1:
        failures += 1
        print(f"FAIL  超长台词的分块没有全部继承 dialogue: {chunks}")
    else:
        print(f"PASS  超长台词拆成 {len(dialogue_chunks)} 块，都标成了 dialogue")

    over = [seg for seg, _ in chunks if tp.width(seg) > tp.MAX_SEGMENT_WIDTH]
    if over:
        failures += 1
        print(f"FAIL  片段超过 MAX_SEGMENT_WIDTH={tp.MAX_SEGMENT_WIDTH}: {over}")
    else:
        print(f"PASS  所有片段都在 MAX_SEGMENT_WIDTH={tp.MAX_SEGMENT_WIDTH} 以内")

    return failures


HANDOFF_CASES = [
    "「你走吧。」",
    "『住手！』他喊道。",
    "他说：「按你在广告上写的，预交三个月房租哈，另外饭费要多少？」",
    "（他站在门口……）",                      # 尾部落单的 ）
    "他读了《漫长的季节》。",
    "【提示】他站在门口。",
    "他说（小声地）然后走了。",
    "「他站在门口……」犹豫了很久。",
    "「他站在门口——」他愣住了。",
    "「他站在门口『很久』……」",
    "——「你走吧。」",
    "他喊：「快走！？」",
]


# 全是省略号的对话：SENTENCE_ENDINGS 把 … 当句末标点，不打包的话会被切成
# 一堆 3 个字的碎片。实测这种段落 rtf = 1.084（合成比朗读还慢），Koodo 的缓冲
# 必然被抽干然后卡死。打包后 rtf = 0.745。
FRAGMENT_PARAGRAPH = (
    "当下郝仁就把前两天那个幻想侵蚀现实的噩梦大略讲了一遍，渡鸦这次难得很安静，"
    "直到郝仁把话说完，她那边才传来一阵可疑的稀里呼噜声：“哦……吸溜……"
    "你做了个怪梦？吸溜……那情况可能确实有点复杂了……吸溜，等会你过来一趟吧吸溜……”"
)

# rtf ≈ 0.746 + 1.147 / 字数（见 TextProcessor._pack_segments 的说明）。
# 留一点余量，要求每段的预估 rtf 不超过这个值。
RTF_CEILING = 0.95


def check_no_fragment_swarm():
    """碎片段是 rtf 杀手，打包之后不能再出现一堆小碎片。

    合成速度只要超过朗读速度（rtf >= 1），Koodo 的缓冲就一定会被抽干，
    表现是有声书读到某句就整个停住、连进度都丢了。
    """
    from tts_engine import TTSEngine

    tp = TextProcessor()
    failures = 0
    segs = tp.split_for_tts(TTSEngine.clean_text(FRAGMENT_PARAGRAPH))
    worst_name, worst_rtf = None, 0.0
    for seg in segs:
        chars = tp.width(seg) / 2
        rtf = 0.746 + 1.147 / chars if chars else 99.0
        if rtf > worst_rtf:
            worst_name, worst_rtf = seg, rtf
    if worst_rtf > RTF_CEILING:
        failures += 1
        print(f"FAIL  有片段预估 rtf={worst_rtf:.3f} 超过 {RTF_CEILING}，"
              f"合成会追不上朗读: {worst_name!r}")
    else:
        print(f"PASS  {len(segs)} 段，最差片段预估 rtf={worst_rtf:.3f}（上限 {RTF_CEILING}）")

    # 打包不能丢字符
    joined = "".join(segs)
    expected = re.sub(r"\s+", "", TTSEngine.clean_text(FRAGMENT_PARAGRAPH))
    if re.sub(r"\s+", "", joined) != expected:
        failures += 1
        print("FAIL  打包丢字符了")
    else:
        print("PASS  打包不丢字符")

    over = [s for s in segs if tp.width(s) > tp.MAX_SEGMENT_WIDTH]
    if over:
        failures += 1
        print(f"FAIL  打包后有片段超过 MAX_SEGMENT_WIDTH={tp.MAX_SEGMENT_WIDTH}: {over}")
    else:
        print(f"PASS  打包后所有片段仍在 MAX_SEGMENT_WIDTH={tp.MAX_SEGMENT_WIDTH} 以内")
    return failures


def check_pause_punctuation():
    """带停顿语义的标点必须自己降级成逗号，指望不上 genie。

    ChineseG2P 的 pattern_filter 只保留 ! ? … , . -，其余非汉字字符整段删掉。
    破折号连同它该有的停顿一起消失（"你说这个啊——普通血族" 会被念成
    "你说这个啊普通血族"，两个分句黏在一起）。它的 PUNCTUATION_REPLACEMENTS
    里虽然写了 "—": "-"，但破折号在映射之前就被 TextNormalizer 吃掉了。
    """
    from tts_engine import TTSEngine
    from genie_tts.G2P.Chinese.ChineseG2P import ChineseG2P

    g2p = ChineseG2P()
    failures = 0
    for text in ["你说这个啊——普通血族确实不喜欢",
                 "他愣了一下——然后笑了",
                 "他说：“哦——原来是这样。”"]:
        cleaned = TTSEngine.clean_text(text)
        text_clean, _, _, _ = g2p.process(cleaned)
        if "," not in text_clean:
            failures += 1
            print(f"FAIL  破折号的停顿丢了: {text!r} -> G2P {text_clean!r}")
    if not failures:
        print("PASS  破折号降级成逗号，停顿保住了")
    return failures


def check_tts_handoff():
    """交给 genie 之前必须把不发音的引号和括号去掉。

    现在 genie.tts 用的是 split_sentence=False，genie 自己的 TextSplitter 不再
    参与，这一条主要是防御性的——万一以后有人把 split_sentence 改回 True，
    下面这些用例能立刻报出来：genie 的 TextSplitter 只认 “”‘’"' 这几种成对
    符号，中文直角引号和各类括号一概不认，它会把 「你走吧。」 切成 「你走吧。
    和 」，而落单的 」 又被 get_effective_len 当成宽度 2 的"内容"，于是这个纯
    标点片段被当成正经句子送去合成——实测能凭空生成 7.28 秒气声。
    """
    from genie_tts.Utils.TextSplitter import TextSplitter
    from tts_engine import UNSPOKEN_MARKS, SPEAKABLE

    tp = TextProcessor()
    splitter = TextSplitter()
    failures = 0
    for text in HANDOFF_CASES:
        for seg in tp.split_for_tts(text):
            cleaned = seg.translate(UNSPOKEN_MARKS).strip()
            if not cleaned or not SPEAKABLE.search(cleaned):
                continue          # 片段级闸门会拦下，不会送去合成
            for sub in splitter.split(cleaned):
                if not SPEAKABLE.search(sub):
                    failures += 1
                    print(f"FAIL  genie 会把纯标点片段 {sub!r} 当成句子合成（来自 {seg!r}）")
    if not failures:
        print(f"PASS  {len(HANDOFF_CASES)} 个含引号/括号的用例都不会切出纯标点片段")
    return failures


def show_passage():
    tp = TextProcessor()
    print("\n--- 整段切分效果（用来听感调 MIN_QUOTE_SEGMENT_WIDTH）---")
    for i, (seg, style) in enumerate(tp.split_for_tts_with_style(PASSAGE)):
        tag = "台词" if style == D else "旁白"
        print(f"  {i:2d} [{tag}] {seg}")


if __name__ == "__main__":
    failures = (check_cases() + check_invariants() + check_no_fragment_swarm()
                + check_pause_punctuation() + check_tts_handoff())
    show_passage()
    print(f"\n{'全部通过' if failures == 0 else f'{failures} 项失败'}")
    sys.exit(1 if failures else 0)
