# Koodo Reader TTS 引擎 (GPT-SoVITS 本地后端)

这是一个为 **Koodo Reader** 提供强大的本地有声书（TTS）功能的后端服务项目。
基于 [Genie-TTS](https://github.com/fluent-tts/genie-tts) 构建，底层使用 **GPT-SoVITS ONNX** 模型进行推理。它可以让你的小说获得接近真人的语音朗读体验，且**全程本地运行、无需断网、保证隐私**。

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

因为 TTS 底层依赖声学与语义大模型，默认的 Git 仓库通常不包含这些几百兆的大文件。
我已经在 `setup` 文件夹中编写了**全自动一键下载脚本**，它会自动连接到 HuggingFace 帮你拉取缺失的核心文件并自动摆放：

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

> 提示：如果你手动往 `characters.json` 中加入了新配置，或者执行了上边的导入脚本，**无需重启 `start_server.bat` 后端服务**，当你下一次听书切换时，它会自动在内存里热加载新数据！

---

## 🔌 Koodo Reader 插件安装与配置

本项目自带了一个 Koodo 测试插件，你可以：

1. 打开 **Koodo Reader**。
2. 进入 Koodo 设置，找到 **插件系统 / 扩展模式**（或按提示向内侧拖放本地插件）。
3. 导入本仓库目录下的 koodo_plugin/koodo_tts_plugin.json 文件进行安装。
4. 在有声书或段落界面，选中小说文本文本点击“听书”。

> 插件只是负责“发请求”的前端。真正朗读文字、进行断句的都是这个 Python 引擎本身。

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
