# 中国上市公司业绩说明会管理层回复情感分析

本仓库提供一项关于**中国上市公司业绩说明会管理层回复情感分类**研究的代码与复现材料。

本研究基于人工标注的管理层回复数据，构建中文金融文本情感测量方法，并比较词典方法、预训练语言模型以及大语言模型在管理层回复情感分类任务中的表现。

\---

## 1\. 研究概述

本研究聚焦于中国上市公司业绩说明会问答环节中的**管理层回复**，并将情感测量构建为二分类任务。

每一条完整的管理层回复作为一个观测样本，并对应一个情感标签：

* `0` = 负面
* `1` = 正面

本仓库提供研究中使用的主要模型代码、训练代码、评价代码及相关复现材料。

\---

## 2\. 数据集

研究使用 **10,000 条人工标注的管理层回复**。

样本类别分布如下：

|标签|情感类别|样本数量|
|-|-|-:|
|0|负面|2,674|
|1|正面|7,326|
|**合计**||**10,000**|

本研究采用\*\*回复层面（response-level）\*\*的标注方式，即一条完整的管理层回复对应一个情感标签，而不是对单独句子进行标注。

### 数据格式

模型代码要求输入 CSV 文件至少包含以下两个字段：

```text
sentence
label
```

其中：

* `sentence`：完整的管理层回复文本；
* `label`：二分类情感标签。

\---

## 3\. 情感分类方法

本仓库包含四类情感测量方法。

### 3.1 词典方法

研究采用两种金融情感词典：

* Loughran–McDonald（LM）金融情感词典；
* 中文金融情感词典（DU）。

相关词典文件及评价代码位于：

```text
Dictionary/
```

\---

### 3.2 StructBERT

研究采用中文 StructBERT 作为预训练语言模型方法。

仓库提供：

* 预训练 StructBERT 的基准评价；
* StructBERT 微调程序；
* 微调后 StructBERT 的评价程序。

相关代码位于：

```text
StructBert/
```

\---

### 3.3 FinBERT

研究同时采用 FinBERT-tone-Chinese 进行金融文本情感分类。

原始 FinBERT 模型采用三分类情感体系。为了与本研究的人工标注任务保持一致，模型被调整为二分类任务：

```text
0 = 负面
1 = 正面
```

FinBERT 的微调及五折交叉验证程序位于：

```text
FinBert/
```

\---

### 3.4 DeepSeek

研究还将 DeepSeek 作为额外的比较方法。

相关代码位于：

```text
Deepseek/
```

仓库不会提供 API Key。

运行 DeepSeek 相关程序时，需要使用者自行配置 API 凭证，并通过环境变量提供。

\---

## 4. 评价指标

研究采用多种评价指标衡量模型分类性能：

* Accuracy
* Balanced Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* ROC-AUC
* Confusion Matrix

由于数据集中正负类别存在一定程度的不平衡，因此研究同时报告 Macro Precision、Macro Recall 和 Macro F1，以避免仅依赖 Accuracy 所造成的评价偏差。

此外，采用 ROC-AUC 作为与分类阈值无关的补充评价指标。

\---

## 5. 仓库结构

```text
Chinese-Earnings-Call-Sentiment/
│
├── README.md
├── requirements.txt
│
├── StructBert/
│   ├── StructBert\_baseline\_evaluate.py
│   ├── Structbert\_finetuning.py
│   └── StructBert\_finetuned\_evaluate.py
│
├── FinBert/
│   ├── finbert-tone-chinese\_finetuning.py
│   ├── train\_cv.py
│   ├── custom\_trainer.py
│   └── utils.py
│
├── Dictionary/
│   ├── Dictionary\_evaluate.py
│   ├── lm\_neg.txt

│   ├── lm\_pos.txt

│   ├── du\_neg.txt
│   └── du\_pos.txt
│
└── Deepseek/
    └── Deepseek\_evaluate.py
```

仓库按照不同情感测量方法进行组织，同时尽可能保留研究实验中实际使用的原始可执行代码。

\---

## 6. 实验环境

研究实验运行于以下环境：

```text
Operating System: Ubuntu 22.04
Python: 3.12
CUDA: 12.8.1
PyTorch: 2.10.0
Transformers: 5.8.1
Datasets: 4.8.4
Scikit-learn: 1.9.0
Pandas: 2.3.3
NumPy: 2.5.1
```

主要 Python 依赖包及版本见：

```text
requirements.txt
```

安装依赖：

```bash
pip install -r requirements.txt
```

\---

## 7. 联系方式

如对复现材料、研究方法或代码实现存在疑问，可以通过 GitHub Issue 与作者联系，或联系对应研究论文的作者。

