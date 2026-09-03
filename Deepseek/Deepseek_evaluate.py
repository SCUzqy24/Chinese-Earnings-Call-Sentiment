# ============================================================
# DeepSeek Zero-shot Sentiment Classification
# ============================================================

import os
import json
import time

import numpy as np
import pandas as pd

from tqdm import tqdm

from openai import OpenAI

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ============================================================
# API KEY
# ============================================================

API_KEY = "your_deepseek_api_key"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# ============================================================
# 基础参数
# ============================================================

MODEL_NAME = "deepseek-v4-flash"     # 当前对应 DeepSeek-V4 API

DATA_PATH = "data.csv"

TEXT_COLUMN = "sentence"

LABEL_COLUMN = "label"

OUTPUT_DIR = "DeepSeek_Results"

BATCH_SIZE = 20

TEMPERATURE = 0

MAX_RETRY = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 读取数据
# ============================================================

print("=" * 65)
print("DeepSeek Zero-shot Sentiment Classification")
print("=" * 65)

print("\nLoading Dataset...")

df = pd.read_csv(DATA_PATH)

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

print(f"Total Samples : {len(df)}")

print("\nClass Distribution")

print(df[LABEL_COLUMN].value_counts())

# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你是一名金融学论文中的人工标注员。

你的任务是严格按照金融学论文中的人工标注规则，对上市公司业绩说明会管理层回复进行二分类。

========================
分类目标
========================

判断该回答是否向投资者释放了积极经营信息（Positive Financial Tone）。你的分类标准应尽可能接近金融学研究中的人工标注，而不是普通自然语言情感分析。

Positive 的比例通常明显高于 Negative。

只要回答中包含任何能够提高投资者对公司未来经营、竞争优势、研发能力、市场地位、战略布局、股东回报、成长能力、产生积极判断的信息，
即使没有披露利润增长，也判定为Positive。

只有回答明确包含经营恶化、风险增加、利润下降、收入下降、重大不确定性，才判定为Negative。

如果只是介绍事实、介绍产品、介绍项目、介绍研发、介绍规划，属于信息增量默认判定Positive。
========================
输出格式
========================

对于每条文本，请输出一个JSON数组(JSON Array)。

例如：

[
 {
   "id":1,
   "label":1,
   "positive_probability":0.91
 },
 {
   "id":2,
   "label":0,
   "positive_probability":0.08
 }
]

不要输出多个独立JSON对象。

不要输出JSON Lines。

整个回复必须是一个JSON数组。
要求：

label 只能是：

0

或

1

positive_probability 为属于 Positive 的概率。

取值范围：

0~1

不要输出解释。

不要输出 Markdown。

不要输出任何其它内容。

如果无法判断，不要输出 0.50。

请选择更符合上述人工标注规则的一侧。


"""

# ============================================================
# 构造Prompt
# ============================================================

def build_prompt(batch_df):

    records = []

    for idx, row in batch_df.iterrows():

        records.append({

            "id": int(idx),

            "text": row[TEXT_COLUMN]

        })

    return json.dumps(
        records,
        ensure_ascii=False,
        indent=2
    )

# ============================================================
# JSON解析
# ============================================================

def parse_json(text):

    text = text.strip()

    # 去Markdown
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()

    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # ---------------------------------------------------
    # 情况1：JSON数组
    # ---------------------------------------------------

    if text.startswith("["):

        return json.loads(text)

    # ---------------------------------------------------
    # 情况2：JSON Lines
    # ---------------------------------------------------

    lines = [
        x.strip()
        for x in text.split("\n")
        if x.strip()
    ]

    try:

        return [
            json.loads(x)
            for x in lines
        ]

    except:

        pass

    # ---------------------------------------------------
    # 情况3：连续JSON对象
    # {"a":1}{"a":2}
    # ---------------------------------------------------

    import re

    objs = re.findall(r"\{.*?\}", text, flags=re.S)

    if len(objs):

        return [

            json.loads(x)

            for x in objs

        ]

    raise ValueError(f"Cannot parse JSON:\n{text}")

# ============================================================
# 单Batch预测
# ============================================================

def predict_batch(batch_df):

    prompt = build_prompt(batch_df)

    retry = 0

    while retry < MAX_RETRY:

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                temperature=TEMPERATURE,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ]

            )

            content = response.choices[0].message.content

            results = parse_json(content)

            return results

        except Exception as e:

            retry += 1

            print(f"\nRetry {retry}/{MAX_RETRY}")

            print(e)

            time.sleep(3)

    raise RuntimeError("DeepSeek API Failed.")

print("\nInitialization Finished.")
print("=" * 65)
# ============================================================
# Batch Prediction
# ============================================================

print("\nStart Batch Prediction...")
print("=" * 65)

y_true = []
y_pred = []
y_prob = []

prediction_records = []

num_batches = int(np.ceil(len(df) / BATCH_SIZE))

for batch_id in tqdm(range(num_batches)):

    start = batch_id * BATCH_SIZE

    end = min(

        start + BATCH_SIZE,

        len(df)

    )

    batch_df = df.iloc[start:end]

    results = predict_batch(batch_df)

    # --------------------------------------------------------
    # 防止返回数量错误
    # --------------------------------------------------------

    if len(results) != len(batch_df):

        print("\nReturned Samples:")

        print(len(results))

        print("Expected Samples:")

        print(len(batch_df))

        raise RuntimeError("Batch size mismatch.")

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    for (_, row), result in zip(batch_df.iterrows(), results):

        true_label = int(row[LABEL_COLUMN])

        pred_label = int(result["label"])

        prob = float(result["positive_probability"])

        # 概率修正

        prob = max(

            0.0,

            min(

                1.0,

                prob

            )

        )

        y_true.append(true_label)

        y_pred.append(pred_label)

        y_prob.append(prob)

        prediction_records.append({

            "sentence": row[TEXT_COLUMN],

            "true_label": true_label,

            "pred_label": pred_label,

            "positive_probability": prob

        })

# ============================================================
# 保存预测结果
# ============================================================

prediction_df = pd.DataFrame(

    prediction_records

)

prediction_path = os.path.join(

    OUTPUT_DIR,

    "prediction.csv"

)

prediction_df.to_csv(

    prediction_path,

    index=False,

    encoding="utf-8-sig"

)

print("\nPrediction Finished.")

y_true = np.array(y_true)

y_pred = np.array(y_pred)

y_prob = np.array(y_prob)

print(f"\nPrediction saved to:")

print(prediction_path)

# ============================================================
# 检查预测类别分布
# ============================================================

print("\nPrediction Distribution")

print(

    prediction_df["pred_label"].value_counts()

)

print("\nGround Truth Distribution")

print(

    prediction_df["true_label"].value_counts()

)

print("=" * 65)
# ============================================================
# Part 3
# Evaluation Metrics
# ============================================================

print("\nCalculating Evaluation Metrics...")
print("=" * 65)

# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

# ============================================================
# Balanced Accuracy
# ============================================================

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

# ============================================================
# Macro Precision / Recall / F1
# ============================================================

macro_precision, macro_recall, macro_f1, _ = \
    precision_recall_fscore_support(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

)

# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(

    y_true,

    y_prob

)

# ============================================================
# 输出结果
# ============================================================

print("\n================ Final Results ================\n")

print(f"Accuracy              : {accuracy:.4f}")

print(f"Balanced Accuracy     : {balanced_acc:.4f}")

print(f"Macro Precision       : {macro_precision:.4f}")

print(f"Macro Recall          : {macro_recall:.4f}")

print(f"Macro F1              : {macro_f1:.4f}")

print(f"ROC-AUC               : {roc_auc:.4f}")

# ============================================================
# 保存Metrics
# ============================================================

metrics_df = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Balanced Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "ROC-AUC"

    ],

    "Value":[

        accuracy,

        balanced_acc,

        macro_precision,

        macro_recall,

        macro_f1,

        roc_auc

    ]

})

metrics_path = os.path.join(

    OUTPUT_DIR,

    "metrics.xlsx"

)

metrics_df.to_excel(

    metrics_path,

    index=False

)

print("\nMetrics saved to:")

print(metrics_path)

print("=" * 65)
# ============================================================
# Part 4
# ROC Curve + Confusion Matrix
# ============================================================

print("\nDrawing ROC Curve...")
print("=" * 65)

# ============================================================
# ROC Curve
# ============================================================

fpr, tpr, thresholds = roc_curve(

    y_true,

    y_prob

)

plt.figure(figsize=(7,6))

plt.plot(

    fpr,

    tpr,

    linewidth=1.2,

    label=f"ROC(AUC = {roc_auc:.4f})"

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

    linewidth=1.2,

    color="gray"

)

plt.xlim(0,1)

plt.ylim(0,1.05)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.grid(alpha=0.3)

roc_path = os.path.join(

    OUTPUT_DIR,

    "roc_curve.png"

)

plt.savefig(

    roc_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("ROC Curve Saved.")

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=["Negative","Positive"]

)

fig, ax = plt.subplots(

    figsize=(6,5)

)

disp.plot(

    cmap="Blues",

    ax=ax,

    colorbar=True,

    values_format="d"

)

plt.title("Confusion Matrix")

cm_path = os.path.join(

    OUTPUT_DIR,

    "confusion_matrix.png"

)

plt.savefig(

    cm_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("Confusion Matrix Saved.")

# ============================================================
# 保存ROC数据
# ============================================================

roc_df = pd.DataFrame({

    "FPR":fpr,

    "TPR":tpr,

    "Threshold":thresholds

})

roc_data_path = os.path.join(

    OUTPUT_DIR,

    "roc_data.xlsx"

)

roc_df.to_excel(

    roc_data_path,

    index=False

)

# ============================================================
# 输出保存位置
# ============================================================

print("\nResults Saved Successfully.")

print("="*65)

print("prediction.csv")

print("metrics.xlsx")

print("roc_curve.png")

print("confusion_matrix.png")

print("roc_data.xlsx")

print("="*65)

print("\nFinished.")