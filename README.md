# NLP 课程实验

自然语言处理课程实验项目，包含中文分词、词向量与句子相似度、CNN/RNN 文本分类、AI 智能问答系统四个实验。

## 环境依赖

| 包 | 版本 | 用途 |
|---|---|---|
| Python | 3.10+ | 运行环境 |
| PyTorch | 2.12.0 | 实验1/2/3 模型训练与推理 |
| NumPy | 2.1.3 | 数值计算 |
| scikit-learn | 1.6.1 | 实验2 余弦相似度 |
| Flask | 3.1.0 | 智能问答系统 Web 服务 |
| OpenAI | 2.41.0 | DeepSeek API 调用 |
| python-dotenv | 1.1.0 | 从 .env 加载环境变量 |
| jieba | 0.34 | 中文分词（离线法律问答） |

一键安装：

```bash
pip install torch numpy scikit-learn flask openai python-dotenv jieba
```

## 项目结构

```
├── 实验1_中文分词/          # 基于前馈神经网络的中文分词
│   └── ffnn.py
├── 实验2_词向量与句子相似度/  # Skip-Gram 词向量训练与句子相似度计算
│   └── wordvec.py
├── 实验3_CNN和RNN文本分类/   # CNN / BiLSTM+Attention 文本分类
│   ├── cnn.py
│   └── rnn.py
├── AI智能问答系统/           # Flask + DeepSeek API 智能问答系统
│   ├── app.py               # Flask 主应用
│   ├── qa_core.py           # 问答核心逻辑（在线/离线模式）
│   ├── local_qa.py          # 离线引擎（数学计算 + 法律问答）
│   ├── dsapi.py             # DeepSeek API 调用
│   ├── db.py                # SQLite 会话持久化
│   ├── config.py            # API 配置（从 .env 读取）
│   ├── .env                 # API 密钥（不上传，需自行创建）
│   ├── law_merged.json      # 法律知识库
│   ├── templates/index.html # 前端页面
│   └── static/              # 前端静态资源
└── icwb2-data/              # SIGHAN Bakeoff 中文分词评测数据集
```

## 实验 1：中文分词

基于前馈神经网络（FFNN）的字符级中文分词，采用 BIES 标注方案。

- **模型**：Embedding → FC → ReLU → Dropout → FC
- **标注**：B（词首）/ I（词中）/ E（词尾）/ S（单字词）
- **数据**：MSR 分词数据集（icwb2-data）
- **评估**：Precision / Recall / F1

```bash
cd 实验1_中文分词
python ffnn.py
```

## 实验 2：词向量与句子相似度

基于 Skip-Gram + 负采样的词向量训练，支持词相似度和句子相似度计算。

- **模型**：Skip-Gram（中心词 + 上下文词 + 负采样）
- **训练**：批处理 SGD，负采样频率按 0.75 次方平滑
- **功能**：词向量导出、词相似度查询、句子向量（均值池化）相似度计算

```bash
cd 实验2_词向量与句子相似度
python wordvec.py
```

## 实验 3：CNN 和 RNN 文本分类

基于 ICWB2 多语料库来源（MSR/PKU/CityU）构建三分类任务，对比 CNN 与 BiLSTM 模型。

### CNN 文本分类

- **模型**：多尺度卷积核 (2,3,4) + BatchNorm + AdaptiveMaxPool + SelfAttention
- **训练**：AdamW + CosineAnnealing + EarlyStopping + 类别加权损失

```bash
cd 实验3_CNN和RNN文本分类
python cnn.py
```

### RNN 文本分类

- **模型**：BiLSTM + Attention + FC
- **训练**：AdamW + CosineAnnealing + EarlyStopping + 梯度裁剪

```bash
python rnn.py
```

## AI 智能问答系统

基于 Flask 的 Web 问答系统，支持在线（DeepSeek API）和离线（本地引擎）两种模式。

### 功能特性

- **多会话管理**：创建/切换/删除对话，自动生成标题
- **在线模式**：调用 DeepSeek API，流式输出
- **离线模式**：数学计算 + 法律问答（两阶段检索）
  - 数学：支持中文数字、四则运算、幂运算、三角函数
  - 法律：基于知识图谱 + 同义词扩展的两阶段检索（咨询回答 + 法条引用）
- **文档上传**：上传文档作为上下文注入
- **系统提示词**：自定义 System Prompt

### 配置

在 `AI智能问答系统/` 目录下创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的API密钥
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
```

> `.env` 文件已被 `.gitignore` 排除，不会上传到 GitHub。

### 启动

```bash
cd AI智能问答系统
python app.py
```

访问 `http://127.0.0.1:7860` 即可使用。
