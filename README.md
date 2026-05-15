# Koodo Reader TTS 引擎 (GPT-SoVITS 本地后端)

这是一个为 **Koodo Reader** 提供强大的本地有声书（TTS）功能的后端服务项目。
基于 [Genie-TTS](https://github.com/fluent-tts/genie-tts) 构建，底层使用 **GPT-SoVITS ONNX** 模型进行推理。它可以让你的小说获得接近真人的语音朗读体验，且**全程本地 CPU 高效运行、无需断网、保证隐私**。

## 🌟 核心特性

- **GPT-SoVITS ONNX 推理**: 本地高性能语音合成，支持动态热加载模型。
- **强大的文本清洗**: 自动过滤零宽字符、排版空格，并优化连续标点和冒号分号导致的 TTS 崩溃/电音问题。
- **极致的 CPU 优化**: 专为 AMD Z1 Extreme (AVX-512) 等现代 CPU 优化，所有推理全部基于 CPU，响应速度极快且不占显存。

---

## 🚀 快速部署指南 (Windows / PC 端)

按照以下简单步骤，即可在任意一台 PC 上快速部署本服务。

### 1. 克隆代码仓库

```bash
git clone <你的仓库地址>
cd audiobook
```

### 2. 初始化 Python 虚拟环境与安装依赖

**要求**: 必须安装 Python 3.10 或更高版本。

为了解决在不同电脑上可能因为缺少微软 C++ 编译器而导致 `jieba_fast` 安装失败的问题，我们提供了一键自动配置环境的脚本：

1. 打开项目内的 `setup` 文件夹。
2. 双击运行 **`setup_venv.bat`**。
3. 等待脚本自动创建对应的 `.venv-genie` 虚拟环境，并安装好所有底层依赖（内置绕过 C++ 编译器限制的修复程序）。

### 3. 一键下载必要的模型文件

因为 TTS 底层依赖声学模型，默认的 GitHub 仓库不包含这些大文件。
运行自动脚本连接 HuggingFace 拉取：

```bash
# 确保在激活了虚拟环境的终端中，运行以下命令：
python setup/download_models.py
```

### 4. 角色管理与导入 (支持 GPT-SoVITS 转换)

本服务完全由根目录的 `characters.json` 控制音色和路径，这意味着**增加新角色再也不需要修改 Python 代码！**

- 我们提供了一个万能的导入脚本 `setup/import_character.py`。
- 如果你有自己训练的 Torch 格式权重 (`.ckpt` 和 `.pth`)，它能一键帮你完成 `Torch -> ONNX转换`、`路径移动`、并自动将其写入并激活到系统的 `characters.json` 里。
- 例：
  ```bash
  python setup/import_character.py --id feibi --name "菲比" --ckpt C:\Downloads\feibi.ckpt --pth C:\Downloads\feibi.pth --audio C:\Downloads\speech.wav --text "测试音频内容"
  ```
  完成之后，系统立刻就能识别出名叫“菲比”的角色。

---

## 🔌 Koodo Reader 插件安装与配置

本项目自带了一个 Koodo 测试插件，你可以：

1. 打开 **Koodo Reader**。
2. 进入 Koodo 设置，找到 **插件系统 / 扩展模式**。
3. 导入本仓库目录下的 `koodo_plugin/koodo_tts_plugin.json` 文件进行安装。
4. 听书时勾选所需的角色（例如：`feibi`）即可体验高质量的语音朗读功能。

---

## 🛠️ 常见问题排错 (FAQ)

### 安装依赖时抛出 `Failed building wheel for jieba_fast` / `Microsoft Visual C++ 14.0 or greater is required.`

这是因为底层环境缺失必要的 C++ 编译环境。
**解决方法**：无需手动排查，只需双击运行本项目提供的 **`setup/setup_venv.bat`**。该脚本会自动创建虚拟环境、安装核心包，并在安装 `genie-tts` 前注入伪装补丁以直接绕过编译限制。

---

## ▶️ 启动与测试服务

**推荐运行方式：**
环境和模型准备完毕后，双击项目根目录下的 **start_server.bat** 即可。

- Koodo 插件通过它唤醒时会**静默后台运行**，没有闪烁烦人的终端黑窗。
- **自动防空载**：如果 Koodo 软件关了，超过 15 分钟无人听书，后台服务就会自动安全退出、清空被占用的内存和显存。

_(如果遇到问题，你可以手动排错执行: python -m uvicorn app:app --host 127.0.0.1 --port 8000)_

**服务自检：**
服务启动后，能在浏览器直接打开验证：
👉 [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status)

如果看到屏幕返回 "status": "ok"，就可以放心地去 Koodo Reader 听小说了！
