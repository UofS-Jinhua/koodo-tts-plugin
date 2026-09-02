import os
import shutil
from huggingface_hub import snapshot_download

# 下载目标始终是项目根目录，而不是当前工作目录，
# 这样从 setup/ 里直接运行本脚本也不会把模型下到错误的地方。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def download_models():
    print("=======================================")
    print("=  Genie-TTS 本地模型自动一键下载脚本 =")
    print("=======================================")
    print("正在从 HuggingFace 仓库(High-Logic/Genie)获取文件，模型较大，需要一段时间...\n")

    # 1. 下载基础部分 (GenieData)
    print("-> 1. 开始下载基础数据 (GenieData/)...")
    # 注意：不要再传 local_dir_use_symlinks，该参数已被 huggingface_hub 移除，
    # 传了会直接 TypeError。新版默认就是复制真实文件，不建符号链接。
    snapshot_download(
        repo_id="High-Logic/Genie",
        allow_patterns=["GenieData/*", "GenieData/**/*"],
        local_dir=PROJECT_ROOT,
    )
    print("   [基础数据下载完成]")

    # 2. 下载 RoBERTa 语义模型部分
    print("\n-> 2. 开始下载 RoBERTa 语义特征模型...")
    snapshot_download(
        repo_id="High-Logic/Genie",
        allow_patterns=["GenieData(Optional)/RoBERTa/*"],
        local_dir=PROJECT_ROOT,
    )

    # 3. 将 RoBERTa 移动到正确的目录 GenieData/RoBERTa/
    src_roberta = os.path.join(PROJECT_ROOT, "GenieData(Optional)", "RoBERTa")
    dst_roberta = os.path.join(PROJECT_ROOT, "GenieData", "RoBERTa")
    
    if os.path.exists(src_roberta):
        print(f"\n-> 3. 正在整理文件夹: 将 {src_roberta} 移至 {dst_roberta} ...")
        # 若已存在，则先合并覆盖内部文件
        if not os.path.exists(dst_roberta):
            os.makedirs(dst_roberta, exist_ok=True)
            
        for item in os.listdir(src_roberta):
            s = os.path.join(src_roberta, item)
            d = os.path.join(dst_roberta, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
                
        # 清理临时下载的(Optional)文件夹
        shutil.rmtree(os.path.join(PROJECT_ROOT, "GenieData(Optional)"))
        print("   [整理完成！]")
    else:
        print("   [未找到临时下载的 RoBERTa 文件夹，可能已经处于正确目录。]")
        
    verify_models()

    print("\n[完成] 全部基础模型下载成功并已归位！")
    print("  接下来你只需要确保你的定制音色(CharacterModels)齐全，即可开始使用。")


# 缺哪个都不会在下载阶段报错，而是等到合成时才以各种奇怪的方式失败
# （例如缺 EnglishG2P 时中英混读会静默退回“英文被跳过”），所以这里直接点名检查。
REQUIRED_PATHS = [
    ("GenieData/chinese-hubert-base", "语音编码器 (必需)"),
    ("GenieData/speaker_encoder.onnx", "音色编码器 (v2ProPlus 必需)"),
    ("GenieData/G2P/ChineseG2P", "中文 G2P (必需)"),
    ("GenieData/G2P/EnglishG2P", "英文 G2P (中英混读需要)"),
    ("GenieData/RoBERTa", "中文 BERT 语义特征 (强烈建议)"),
]


def verify_models() -> bool:
    """检查关键模型文件是否齐全。返回 True 表示一个都不缺。"""
    print("\n-> 4. 正在检查模型文件是否齐全 ...")
    missing = []
    for rel_path, label in REQUIRED_PATHS:
        full_path = os.path.join(PROJECT_ROOT, *rel_path.split("/"))
        exists = os.path.exists(full_path)
        print("   [%s] %-32s %s" % ("OK  " if exists else "缺失", rel_path, label))
        if not exists:
            missing.append(rel_path)
    if missing:
        print("\n   [警告] 以下文件没有下载成功，检查网络后重新运行本脚本即可续传：")
        for rel_path in missing:
            print("     - " + rel_path)
    return not missing


if __name__ == '__main__':
    download_models()
