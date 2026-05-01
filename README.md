# Koodo Reader TTS 引擎 (GPT-SoVITS 本地后端 + 情感分析)

这是一个为 **Koodo Reader** 提供强大的本地有声书（TTS）功能的后端服务项目。
基于 [Genie-TTS](https://github.com/fluent-tts/genie-tts) 构建，底层使用 **GPT-SoVITS ONNX** 模型进行推理，并集成 **mDeBERTa-v3** 零样本分类本地 NLP 模型实现动态情感语音切换。它可以让你的小说获得接近真人的语音朗读体验，且**全程本地 CPU 高效运行、无需断网、保证隐私**。

## 🌟 核心特性

- **GPT-SoVITS ONNX 推理**: 本地高性能语音合成，支持动态热加载模型。
- **上下文感知的情感分析**: 自动结合上下文（前置、后置句子），使用 `mDeBERTa-v3-base-mnli-xnli` 本地大语言模型进行零样本分类（开心、伤心、愤怒、平静），精准把握当前句子的情绪。
- **无缝情绪语音切换**: 对同一角色支持在不重新加载底层模型的前提下，通过智能替换参考音频（Reference Audio）实现情绪音色的无缝切换。
- **强大的文本清洗**: 自动过滤零宽字符、排版空格，并优化连续标点和冒号分号导致的 TTS 崩溃/电音问题。
- **极致的 CPU 优化**: 专为 AMD Z1 Extreme (AVX-512) 等现代 CPU 优化，所有推理（TTS + NLP）全部基于 CPU，响应速度极快且不占显存。
- **自包含模型缓存**: NLP 模型会自动下载并隔离存储在项目目录的 `NLP_Model/` 中，保持环境纯净。

---

## 🚀 快速部署指南 (Windows / PC 端)

按照以下简单步骤，即可在任意一台 PC 上快速部署本服务。

### 1. 克隆代码仓库

```bash
git clone <你的仓库地址>
cd audiobook
```

### 2. 初始化 Python 虚拟环境

**要求**: 必须安装 Python 3.10 或更高版本。

在项目根目录下打开终端，执行以下命令创建并激活虚拟环境：

```bash
# 创建名叫 .venv-genie 的虚拟环境 (名称需与启动脚本一致)
python -m venv .venv-genie

# 激活环境
.venv-genie\Scripts\activate

# 安装所需的依赖包
pip install -r requirements.txt
```

### 3. 一键下载必要的 AI 模型文件

因为 TTS 和 NLP 底层依赖声学与语义大模型，默认的 GitHub 仓库不包含这些大文件。
**对于 TTS 模型**，运行自动脚本连接 HuggingFace 拉取：

```bash
# 确保在激活了虚拟环境的终端中，运行以下命令：
python setup/download_models.py
```

**对于 NLP 情感模型（mDeBERTa-v3）**：
NLP 分析器会在第一次启动项目服务端时，自动从 HuggingFace 离线缓存至当前项目路径下的 `NLP_Model/` 文件夹内（约为几百兆），全程不影响系统其他环境变量。

### 4. 角色管理与导入 (支持 GPT-SoVITS 转换)

本服务完全由根目录的 `characters.json` 控制音色和路径，这意味着**增加新角色再也不需要修改 Python 代码！**
- 我们提供了一个万能的导入脚本 `setup/import_character.py`。
- 如果你有自己训练的 Torch 格式权重 (`.ckpt` 和 `.pth`)，它能一键帮你完成 `Torch -> ONNX转换`、`路径移动`、并自动将其写入并激活到系统的 `characters.json` 里。
- 例：
  ```bash
  python setup/import_character.py --id feibi --name "菲比" --ckpt C:\Downloads\feibi.ckpt --pth C:\Downloads\feibi.pth --audio C:\Downloads\speech.wav --text "测试音频内容"
  ```
  完成之后，系统立刻就能识别出名叫“菲比”的角色。

> 提示：如果需要支持某角色的**多情绪音色覆盖**，只需在 `characters.json` 中添加带情绪后缀的别名字段，例如 `"feibi_happy"`、`"feibi_angry"`，引用不同的参考音频即可，主模型无需更改。

---

## 🔌 Koodo Reader 插件安装与配置

本项目自带了一个 Koodo 测试插件，你可以：

1. 打开 **Koodo Reader**。
2. 进入 Koodo 设置，找到 **插件系统 / 扩展模式**。
3. 导入本仓库目录下的 `koodo_plugin/koodo_tts_plugin.json` 文件进行安装。
4. 听书时勾选所需的角色（例如：`feibi`）即可体验带语气感知的语音朗读功能。

---

## ▶️ 启动与测试服务

**推荐运行方式：**
环境和模型准备完毕后，双击项目根目录下的 **start_server.bat** 即可。
- Koodo 插件通过它唤醒时会**静默后台运行**，没有闪烁烦人的终端黑窗。
- **自动防空载**：如果 Koodo 软件关了，超过 15 分钟无人听书，后台服务就会自动安全退出、清空被占用的内存和显存。

*(如果遇到问题，你可以手动排错执行: python -m uvicorn app:app --host 127.0.0.1 --port 8000)*

**服务自检：**
服务启动后，能在浏览器直接打开验证：
👉 [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status)

如果看到屏幕返回 "status": "ok"，就可以放心地去 Koodo Reader 听小说了！
