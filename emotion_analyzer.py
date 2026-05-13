import os
import logging
from transformers import pipeline

# 在导入或创建 pipeline 前锁定 HuggingFace 的缓存目录为当前目录的 NLP_Model 文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_MODEL_DIR = os.path.join(BASE_DIR, "NLP_Model")
os.environ["HF_HOME"] = NLP_MODEL_DIR
os.environ["TRANSFORMERS_CACHE"] = NLP_MODEL_DIR

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self):
        logger.info("[Emotion] Loading zero-shot classification model (CPU)...")
        try:
            self.classifier = pipeline(
                task="zero-shot-classification", 
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", 
                device=-1,
                model_kwargs={"cache_dir": NLP_MODEL_DIR} # 双重保险，指定下载缓存路径
            )
        except Exception as e:
            logger.error(f"[Emotion] Failed to load model: {e}")
            self.classifier = None
        
        self.candidate_labels = ["开心", "伤心", "愤怒", "严肃", "平静"]
        self.label_mapping = {
            "开心": "happy",
            "伤心": "sad",
            "愤怒": "angry",
            "严肃": "serious",
            "平静": "neutral"
        }

    def analyze(self, text: str, prev_text: str = "", next_text: str = "") -> str:
        if len(text.strip()) < 3 or not self.classifier:
            return "neutral"
            
        try:
            # 融合上下文作为参考背景，增强零样本分类对隐喻、反讽的理解
            context_text = text
            if prev_text or next_text:
                parts = []
                if prev_text:
                    parts.append(f"前文：{prev_text}")
                parts.append(f"当前句：{text}")
                if next_text:
                    parts.append(f"后文：{next_text}")
                context_text = "。".join(parts)

            result = self.classifier(context_text, self.candidate_labels)
            
            best_label = result['labels'][0]
            best_score = result['scores'][0]
            
            # 对于四分类，0.25 是瞎蒙概率。
            # 文本中如果有“生锈”、“废弃”、“断裂”，NLP 很容易给出 0.4 左右的“伤心”得分。
            # 为了确保语气不过度切换（只在明显的情绪爆发时才换语气），把阈值提高到 0.9
            if best_score < 0.9:
                return "neutral"
                
            return self.label_mapping.get(best_label, "neutral")
            
        except Exception as e:
            logger.error(f"[Emotion] Analysis error: {e}")
            return "neutral"

# Global singleton instance
analyzer = EmotionAnalyzer()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(analyzer.analyze("今天去游乐园玩，真是太开心啦！")) # Expected: happy
    print(analyzer.analyze("为什么大家都不理我，我做错什么了吗...")) # Expected: sad
    print(analyzer.analyze("你给我滚开，别拿脏手碰我！")) # Expected: angry
    print(analyzer.analyze("这份报告你今晚必须交给我，不容有失。")) # Expected: serious
    print(analyzer.analyze("他走到窗边，推开了玻璃窗。")) # Expected: neutral