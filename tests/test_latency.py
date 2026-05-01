import time
import os
from emotion_analyzer import analyzer

texts = [
    "浅浅和小泡泡当然不会瞬移，之所以凭空出现，完全是突然玩心高涨的浅浅在捣乱：",
    "浅浅把小泡泡抱回来三次，然后又领着手上沾满了铁锈的小泡泡去不远处找水管洗了把手，"
]

for t in texts:
    start = time.time()
    emo = analyzer.analyze(t)
    end = time.time()
    print(f"Time for '{t[:10]}...': {end - start:.2f}s, Emotion: {emo}")
