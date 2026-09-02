import os
import json
import argparse
import genie_tts as genie

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def convert_and_register_character(
    short_name,
    display_name,
    ckpt_path,
    pth_path,
    ref_audio,
    ref_text,
    language="Chinese",
    desc=""
):
    print("==========================================")
    print(f"=  新角色导入向导 ({display_name}) ")
    print("==========================================\n")

    # 1. 创建基于 v2ProPlus 的专属目录
    base_target_dir = os.path.join(project_root, "CharacterModels", "v2ProPlus", short_name)
    tts_models_dir = os.path.join(base_target_dir, "tts_models")
    os.makedirs(tts_models_dir, exist_ok=True)
    
    print(f"-> 步骤 1: 开始 ONNX 模型格式转换 ...")
    print(f"   来源: {ckpt_path}")
    print(f"   输出: {tts_models_dir}")
    
    try:
        genie.convert_to_onnx(
            torch_ckpt_path=ckpt_path,
            torch_pth_path=pth_path,
            output_dir=tts_models_dir
        )
        print("   [转换完成]")
    except Exception as e:
        print(f"   [转换失败]: {e}")
        print("   -> 请确保你提供了正确的 GPT-SoVITS 训练权重路径。")
        return

    # 2. 将参考音频复制到模型目录下 (为了集中管理), 若参考音频是同名音频则直接覆盖
    import shutil
    final_audio_path = os.path.join(base_target_dir, os.path.basename(ref_audio))
    if os.path.abspath(ref_audio) != os.path.abspath(final_audio_path):
        print(f"-> 步骤 2: 正在复制参考音频到统一目录 ...")
        shutil.copy2(ref_audio, final_audio_path)
    
    # 获取基于项目根目录的相对路径，以写入 JSON
    rel_model_dir = os.path.relpath(tts_models_dir, project_root).replace('\\', '/')
    rel_audio_path = os.path.relpath(final_audio_path, project_root).replace('\\', '/')

    # 3. 注册到 characters.json
    print("-> 步骤 3: 正在更新 characters.json 数据库 ...")
    json_path = os.path.join(project_root, "characters.json")
    
    config = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
    config[short_name] = {
        "name": display_name,
        "desc": desc or f"{display_name} - 本地添加角色",
        "lang": language,
        "type": "custom",
        "model_dir": rel_model_dir,
        "ref_audio": rel_audio_path,
        "ref_text": ref_text
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    print(f"   [注册成功] JSON 文件已更新。")
    print(f"\n[完成] 全新模型已就绪！")
    print(f"   现在后端系统和 Koodo 插件会自动识别【{display_name}】。")
    print(f"   (如果服务器正在运行，不用重启后端，下次切换角色时它会自动拉取最新数据库)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换 GPT-SoVITS Torch 权重为本地 ONNX 服务格式，并自动注册到系统。")
    parser.add_argument("--id", required=True, help="英文短标识，如 (ayaka)")
    parser.add_argument("--name", required=True, help="显示名称，如 (神里绫华)")
    # 注意别搞反：GPT-SoVITS 训练产出里 .ckpt 是 GPT(s1) 权重、.pth 是 SoVITS(s2) 权重
    parser.add_argument("--ckpt", required=True, help=".ckpt 格式的 GPT (s1) 权重路径")
    parser.add_argument("--pth", required=True, help=".pth 格式的 SoVITS (s2) 权重路径")
    parser.add_argument("--audio", required=True, help="角色参考音频 (.wav) 路径")
    parser.add_argument("--text", required=True, help="参考音频对应的文字内容")
    parser.add_argument("--lang", default="Chinese", help="语言 (Chinese/Japanese/English), 默认: Chinese")
    parser.add_argument("--desc", default="", help="角色描述, 可选")

    args = parser.parse_args()

    # 提前校验，避免跑完一半才发现路径打错
    for label, path in [("--ckpt", args.ckpt), ("--pth", args.pth), ("--audio", args.audio)]:
        if not os.path.isfile(path):
            parser.error(f"{label} 指向的文件不存在: {path}")

    convert_and_register_character(
        short_name=args.id,
        display_name=args.name,
        ckpt_path=args.ckpt,
        pth_path=args.pth,
        ref_audio=args.audio,
        ref_text=args.text,
        language=args.lang,
        desc=args.desc
    )
