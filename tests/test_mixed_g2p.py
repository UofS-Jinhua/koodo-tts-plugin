# -*- coding: utf-8 -*-
"""
中英混读补丁的自检脚本（只跑 G2P，不加载声学模型，几秒就能出结果）。

    .venv-genie\Scripts\python.exe tests\test_mixed_g2p.py

用 --tts 参数可以额外跑一次真实合成，输出 tests/mixed_g2p_output.wav：

    .venv-genie\Scripts\python.exe tests\test_mixed_g2p.py --tts
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # GenieData 是相对路径
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import mixed_g2p

CASES = [
    "我今天用ChatGPT写了一段代码，效果很好。",
    "他掏出iPhone，打开Wi-Fi。",
    "GPT-4 的表现比 GPT-3.5 强很多。",
    "她说：don't worry，一切都会好的。",
    "Hello world.",
    "这句话里一个英文都没有，走的还是原来的分支。",
    "AI、CEO、OK 这类缩写也不该被吞掉。",
]

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print("   [FAIL]", message)
    return condition


print("=== 1. 语种分段 ===")
for text in CASES:
    runs = mixed_g2p.split_language_runs(text)
    print(" ", text)
    print("   ", [(k, s) for k, s in runs if s.strip()])

print("\n=== 2. 补丁未启用时：英文被吞掉（复现问题）===")
before = {}
for text in CASES:
    seq, bert = mixed_g2p.get_phones_and_bert.__globals__["_ORIGINAL_GET_PHONES_AND_BERT"](text, language="Chinese")
    before[text] = seq.shape[1]
    print("   %-34s 音素数 %3d  bert %s" % (text[:16], seq.shape[1], bert.shape))
    check(seq.shape[1] == bert.shape[0], "原实现自身音素/BERT 行数不一致: %r" % text)

print("\n=== 3. 补丁启用后 ===")
mixed_g2p.apply()
import genie_tts.Core.Inference as inference_module
import genie_tts.Audio.ReferenceAudio as reference_audio_module

check(mixed_g2p.is_applied(), "GetPhonesAndBert 模块未被patch")
check(inference_module.get_phones_and_bert is mixed_g2p.get_phones_and_bert, "Core.Inference 未被patch")
check(reference_audio_module.get_phones_and_bert is mixed_g2p.get_phones_and_bert, "Audio.ReferenceAudio 未被patch")

for text in CASES:
    seq, bert = inference_module.get_phones_and_bert(text, language="Chinese")
    has_en = any(k == "en" for k, _ in mixed_g2p.split_language_runs(text))
    print("   %-34s 音素数 %3d -> %3d  bert %s" % (text[:16], before[text], seq.shape[1], bert.shape))
    check(seq.shape[1] == bert.shape[0], "音素数与 BERT 行数不一致: %r" % text)
    if has_en:
        check(seq.shape[1] > before[text], "含英文的句子音素数没有增加，英文可能仍被跳过: %r" % text)
    else:
        check(seq.shape[1] == before[text], "纯中文句子的结果被改变了: %r" % text)

print("\n=== 4. 非中文角色应原样转交 ===")
en_seq, en_bert = inference_module.get_phones_and_bert("Hello world.", language="English")
check(en_seq.shape[1] > 0 and en_bert.shape[0] == en_seq.shape[1], "English 角色的结果不正确")
check(not en_bert.any(), "English 角色不应该有非零 BERT 特征")
print("   English: 音素数 %d, bert %s" % (en_seq.shape[1], en_bert.shape))

if "--tts" in sys.argv:
    print("\n=== 5. 真实合成 ===")
    from tts_engine import TTSEngine

    engine = TTSEngine(character="kiana")
    wav = engine.synthesize("我今天用ChatGPT写了一段代码，效果很好。")
    out = os.path.join("tests", "mixed_g2p_output.wav")
    with open(out, "wb") as f:
        f.write(wav)
    print("   已写入 %s (%d 字节)" % (out, len(wav)))
    check(len(wav) > 44, "合成结果为空")

print("\n" + ("通过：全部检查项 OK" if not failures else "失败 %d 项：\n  - %s" % (len(failures), "\n  - ".join(failures))))
sys.exit(1 if failures else 0)
