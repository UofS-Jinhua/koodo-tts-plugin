# Koodo Reader TTS 引擎 (GPT-SoVITS 本地后端)

这是一个为 **Koodo Reader** 提供强大的本地有声书（TTS）功能的后端服务项目。
基于 [Genie-TTS](https://github.com/fluent-tts/genie-tts) 构建，底层使用 **GPT-SoVITS ONNX** 模型进行推理。它可以让你的小说获得接近真人的语音朗读体验，且**全程本地 CPU 高效运行、无需断网、保证隐私**。

## 🌟 核心特性

- **GPT-SoVITS ONNX 推理**: 本地高性能语音合成，支持动态热加载模型。
- **中英混读**: 自动把中文句子里夹带的英文（`ChatGPT`、`Wi-Fi`、`GPT-4`…）路由到英文 G2P，不再被 genie-tts 的中文前端整段丢弃。
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
.venv-genie\Scripts\python.exe setup\download_models.py
```

脚本跑完会自检一次必需的模型目录（含中英混读要用的英文 G2P），缺什么会直接列出来。

### 4. 角色管理与导入 (支持 GPT-SoVITS 转换)

本服务完全由根目录的 `characters.json` 控制音色和路径，这意味着**增加新角色再也不需要修改 Python 代码！**

- 我们提供了一个万能的导入脚本 `setup/import_character.py`。
- 如果你有自己训练的 Torch 格式权重 (`.ckpt` 和 `.pth`)，它能一键帮你完成 `Torch -> ONNX转换`、`路径移动`、并自动将其写入并激活到系统的 `characters.json` 里。
- 例：
  ```bash
  .venv-genie\Scripts\python.exe setup\import_character.py --id ayaka --name "神里绫华" --ckpt C:\Downloads\ayaka.ckpt --pth C:\Downloads\ayaka.pth --audio C:\Downloads\speech.wav --text "测试音频内容"
  ```
  完成之后，系统立刻就能识别出名叫“神里绫华”的角色。（别拿 `--id feibi` 当例子跑，那个 id 已经被 `characters.json` 里的预设角色占用，会被覆盖成自定义角色。）
- ⚠️ **别把两个权重搞反**：GPT-SoVITS 训练产出里 `.ckpt` 是 GPT (s1) 权重、`.pth` 是 SoVITS (s2) 权重。传反了 ONNX 转换会直接失败。

---

## 🔌 Koodo Reader 插件安装与配置

本项目自带了一个 Koodo 测试插件，你可以：

1. 打开 **Koodo Reader**。
2. 进入 Koodo 设置，找到 **插件系统 / 扩展模式**。
3. 导入本仓库目录下的 `koodo_plugin/koodo_tts_plugin.json` 文件进行安装。
4. 听书时勾选所需的角色即可体验高质量的语音朗读功能。

⚠️ **换一台电脑时必须改这个文件**：插件脚本里写死了 `C:\Codes\audiobook\start_server.bat`
和同样的 `cwd`（Koodo 插件只认绝对路径）。仓库放在别的位置时要把这两处改成本机路径，
**并且重算 `scriptSHA256`**，否则 Koodo 会因为校验不过而拒绝加载：

```bash
.venv-genie\Scripts\python.exe -c "import json,hashlib;p='koodo_plugin/koodo_tts_plugin.json';d=json.load(open(p,encoding='utf-8'));d['scriptSHA256']=hashlib.sha256(d['script'].encode()).hexdigest();json.dump(d,open(p,'w',encoding='utf-8'),ensure_ascii=False,indent=2);print(d['scriptSHA256'])"
```

---

## 🛠️ 常见问题排错 (FAQ)

### 安装依赖时抛出 `Failed building wheel for jieba_fast` / `Microsoft Visual C++ 14.0 or greater is required.`

这是因为底层环境缺失必要的 C++ 编译环境。
**解决方法**：无需手动排查，只需双击运行本项目提供的 **`setup/setup_venv.bat`**。
该脚本会先尝试正常安装 `jieba_fast`；只有在编译失败时，才自动改装纯 Python 的 `jieba`，
并用 `create_jieba_shim.py` 在 site-packages 里生成一个名为 `jieba_fast` 的转发模块顶替它，
然后再安装 `requirements.txt` 里的其余依赖。

### 中文句子里夹带的英文被整段跳过

这是 `genie-tts` 上游的行为：它的语言是**按角色**锁死的（`genie.tts()` 没有 `language` 参数），
`GetPhonesAndBert.py` 会把整段文本丢给中/英/日三选一的 G2P，而中文 G2P 里明确写着「移除英文」
（`ChineseG2P.py` 的 `pattern_filter` 只保留汉字和标点，`g2p()` 里还有一句 `pattern_eng.sub("", seg)`）。

**解决方法**：本项目的 `mixed_g2p.py` 已经在运行时替换掉了 `get_phones_and_bert`，
按语种把句子切成中/英片段分别做 G2P 再拼回同一条音素序列（英文片段用零 BERT 特征，
与上游 GPT-SoVITS 的中英混合模式一致），整句仍然只跑一次推理。
`tts_engine.py` 导入时会自动调用 `mixed_g2p.apply()`，无需任何配置；
因为是猴补丁而不是改 `site-packages` 源码，重建虚拟环境后依然有效。

自检（几秒出结果，加 `--tts` 会额外合成一条 wav）：

```bash
.venv-genie\Scripts\python.exe tests\test_mixed_g2p.py --tts
```

---

## ▶️ 启动与测试服务

**推荐运行方式：**
环境和模型准备完毕后，双击项目根目录下的 **start_server.bat** 即可。

- Koodo 插件通过它唤醒时会**静默后台运行**，没有闪烁烦人的终端黑窗。
- **自动防空载**：如果 Koodo 软件关了，超过 15 分钟无人听书，后台服务就会自动安全退出、清空被占用的内存和显存。

_(如果遇到问题，你可以手动排错执行: python -m uvicorn app:app --host 127.0.0.1 --port 8000)_

**另一条路线（可选）：`start_neurolink_server.bat` + `koodo_plugin/neurolink_tts_plugin.json`**
它启动的不是本项目的 genie-tts 后端，而是另一个仓库里完整的 GPT-SoVITS 服务（端口 8002）。
脚本顶部的 `TTS_SERVER_DIR` 是写死的本机路径（`C:\Codes\Neurolink-Node\TTS_Server`），
换机器时要改成那台机器上的实际位置；插件 json 同样要改路径并重算 `scriptSHA256`（方法见上）。
两条路线互相独立，本文档其余部分描述的都是 genie-tts 后端。

**服务自检：**
服务启动后，能在浏览器直接打开验证：
👉 [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status)

如果看到屏幕返回 "status": "ready"，就可以放心地去 Koodo Reader 听小说了！
