import os
import shutil
from huggingface_hub import snapshot_download

def download_models():
    print("=======================================")
    print("=  Genie-TTS 本地模型自动一键下载脚本 =")
    print("=======================================")
    print("正在从 HuggingFace 仓库(High-Logic/Genie)获取文件，模型较大，需要一段时间...\n")

    # 1. 下载基础部分 (GenieData)
    print("-> 1. 开始下载基础数据 (GenieData/)...")
    snapshot_download(
        repo_id="High-Logic/Genie",
        allow_patterns=["GenieData/*", "GenieData/**/*"],
        local_dir=".",
        local_dir_use_symlinks=False
    )
    print("   [基础数据下载完成]")

    # 2. 下载 RoBERTa 语义模型部分
    print("\n-> 2. 开始下载 RoBERTa 语义特征模型...")
    snapshot_download(
        repo_id="High-Logic/Genie",
        allow_patterns=["GenieData(Optional)/RoBERTa/*"],
        local_dir=".",
        local_dir_use_symlinks=False
    )
    
    # 3. 将 RoBERTa 移动到正确的目录 GenieData/RoBERTa/
    src_roberta = os.path.join(".", "GenieData(Optional)", "RoBERTa")
    dst_roberta = os.path.join(".", "GenieData", "RoBERTa")
    
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
        shutil.rmtree(os.path.join(".", "GenieData(Optional)"))
        print("   [整理完成！]")
    else:
        print("   [未找到临时下载的 RoBERTa 文件夹，可能已经处于正确目录。]")
        
    print("\n🎉 全部基础模型下载成功并已归位！")
    print("  接下来你只需要确保你的定制音色(CharacterModels)齐全，即可开始使用。")

if __name__ == '__main__':
    download_models()
