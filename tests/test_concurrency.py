# -*- coding: utf-8 -*-
"""
并发合成的自检脚本：验证长请求不会把短请求堵住几十秒。

    .venv-genie\\Scripts\\python.exe tests\\test_concurrency.py

背景：实测抓到过真实故障——一段长文本被切成 13 个片段，占着锁连续合成
61 秒，同一时间 Koodo 并发预取的 8 个短句请求全部卡到那 61 秒末尾才一起
返回。Koodo 等不到它正要播的那一句，播放就断在那里，而服务端日志看到的
只是"最后全部 200"，没有任何报错。

修法是把锁从"整个请求的所有片段"下沉到"每一次 genie.tts() 调用"，让短
请求可以插在长请求的片段之间。这个测试复现同样的并发场景，确认：
  1. 短句不会被长请求整体拖住——应该在长请求结束前就返回；
  2. 交错执行不会互相串音——每个短句拿到的时长跟它单独跑时基本一致。
"""
import io
import os
import sys
import threading
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from tts_engine import TTSEngine

LONG_TEXT = "".join([
    "然后他们有一个专门管理各个世界的组织叫做时空管理局，",
    "有一个叫做渡鸦的奇怪女人说自己是神，然后找你当她的助手，",
    "综上所述你现在是这个世界唯一能阻止灾难的人，你必须尽快做出选择，",
    "要么留下来面对接下来的一切，要么转身离开再也不回头。",
])
SHORT_TEXTS = ["什么？", "「等等！」", "他愣住了。", "为什么？", "好。"]

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print("   [FAIL]", message)
    return condition


def duration_of(wav_bytes: bytes) -> float:
    if not wav_bytes:
        return 0.0
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def main():
    engine = TTSEngine("kiana")
    results = {}
    results_lock = threading.Lock()

    def run(name, text, delay=0.0):
        if delay:
            time.sleep(delay)
        t0 = time.time()
        wav = engine.synthesize(text)
        elapsed = time.time() - t0
        with results_lock:
            results[name] = (elapsed, duration_of(wav))

    print("=== 基准：短句各自单独跑一次，记录正常时长 ===")
    baseline = {}
    for i, text in enumerate(SHORT_TEXTS):
        engine._cache.clear()
        engine._cache_keys.clear()
        wav = engine.synthesize(text)
        baseline[i] = duration_of(wav)
        print(f"   short-{i}: {baseline[i]:.2f}s   {text}")

    print("\n=== 并发：长段落 + 5 个短句错开发起，短句应远早于长段落返回 ===")
    engine._cache.clear()
    engine._cache_keys.clear()
    threads = [threading.Thread(target=run, args=("long", LONG_TEXT))]
    for i, text in enumerate(SHORT_TEXTS):
        # 错开起步，模拟 Koodo 陆续发出的并发预取，且都晚于长请求到达
        threads.append(threading.Thread(target=run, args=(f"short-{i}", text, 0.3 + i * 0.05)))

    for th in threads:
        th.start()
    for th in threads:
        th.join()

    long_elapsed, long_dur = results["long"]
    print(f"   long: 耗时 {long_elapsed:.1f}s  时长 {long_dur:.2f}s")
    for i in range(len(SHORT_TEXTS)):
        elapsed, dur = results[f"short-{i}"]
        print(f"   short-{i}: 耗时 {elapsed:.1f}s  时长 {dur:.2f}s")

    worst_short_elapsed = max(results[f"short-{i}"][0] for i in range(len(SHORT_TEXTS)))
    check(
        worst_short_elapsed < long_elapsed * 0.8,
        f"短句最长等待 {worst_short_elapsed:.1f}s，接近甚至超过长请求耗时 {long_elapsed:.1f}s"
        "——说明锁粒度又变粗了，短句被长段落整体卡住",
    )

    print("\n=== 串音校验：并发场景下每句的时长应接近它单独跑的基准 ===")
    for i in range(len(SHORT_TEXTS)):
        _, concurrent_dur = results[f"short-{i}"]
        base = baseline[i]
        ratio = concurrent_dur / base if base else 0
        print(f"   short-{i}: 并发 {concurrent_dur:.2f}s / 基准 {base:.2f}s (比值 {ratio:.2f})")
        check(0.5 <= ratio <= 2.0,
              f"short-{i} 并发时长与单独基准偏差过大 (比值 {ratio:.2f})，疑似串音/内容错位")

    print("\n" + ("通过：全部检查项 OK" if not failures else
                   "失败 %d 项：\n  - %s" % (len(failures), "\n  - ".join(failures))))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
