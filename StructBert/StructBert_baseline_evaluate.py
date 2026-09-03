"""
==========================================================
StructBERT Baseline
5-Fold Cross Validation

Model:
iic/nlp_structbert_sentiment-classification_chinese-base

Evaluation Metrics

1. Accuracy
2. Balanced Accuracy
3. Macro Precision
4. Macro Recall
5. Macro F1
6. ROC-AUC

Outputs

1. Fold Prediction
2. ROC Curve
3. Confusion Matrix
4. Mean ROC
5. Summary Excel
==========================================================
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    precision_recall_fscore_support,

    roc_auc_score,

    roc_curve,

    confusion_matrix,

    auc,

    ConfusionMatrixDisplay

)

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# ==========================================================
# 基础参数
# ==========================================================

model_id = "iic/nlp_structbert_sentiment-classification_chinese-base"

data_path = "data.csv"

text_column = "sentence"

label_colunm = "label"

n_splits = 5

random_seed = 42

OUTPUT_DIR = "./output_structbert"

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)


# ==========================================================
# 绘制ROC
# ==========================================================

def plot_roc_curve(

        y_true,

        y_prob,

        save_path,

        title

):
    fpr, tpr, _ = roc_curve(

        y_true,

        y_prob

    )

    auc_value = auc(

        fpr,

        tpr

    )

    plt.figure(figsize=(6, 6))

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"ROC(AUC = {roc_auc:.4f})"

    )

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        linewidth=1.2,

        color="gray"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(title)

    plt.legend(loc="lower right")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(f"ROC Saved -> {save_path}")

    return fpr, tpr, auc_value


# ==========================================================
# 绘制Confusion Matrix
# ==========================================================

def plot_confusion_matrix(

        y_true,

        y_pred,

        save_path,

        labels=("Negative", "Positive")

):
    cm = confusion_matrix(

        y_true,

        y_pred

    )

    fig, ax = plt.subplots(

        figsize=(6, 5)

    )

    image = ax.imshow(

        cm,

        interpolation="nearest",

        cmap=plt.cm.Blues

    )

    plt.colorbar(image)

    ax.set_xticks(np.arange(len(labels)))

    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)

    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Label")

    ax.set_ylabel("True Label")

    ax.set_title("Confusion Matrix")

    threshold = np.max(cm) / 2

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):
            ax.text(

                j,

                i,

                format(cm[i, j], "d"),

                ha="center",

                va="center",

                color="white"

                if cm[i, j] > threshold

                else "black"

            )

    plt.tight_layout()

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(f"Confusion Matrix Saved -> {save_path}")


# ==========================================================
# 加载StructBERT模型（仅加载一次）
# ==========================================================

print("\nLoading StructBERT...")

semantic_cls = pipeline(
    Tasks.text_classification,
    model=model_id
)

print("Model Loaded.")
# ==========================================================
# 读取数据
# ==========================================================

print("\nLoading Dataset...")

df = pd.read_csv(data_path)

df = df[["sentence", "label"]].dropna()

df["label"] = df["label"].astype(int)

print(f"Total Samples : {len(df)}")

print("\nClass Distribution")

print(df["label"].value_counts())

# ==========================================================
# 五折交叉验证
# ==========================================================

skf = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

# ==========================================================
# 保存各折评价指标
# ==========================================================

all_acc = []
all_bal_acc = []
all_prec = []
all_rec = []
all_f1 = []
all_auc = []

# ==========================================================
# 保存ROC信息
# ==========================================================

roc_infos = []

# ==========================================================
# 保存所有预测结果（用于最终混淆矩阵）
# ==========================================================

all_y_true = []
all_y_pred = []

# ==========================================================
# 创建输出目录
# ==========================================================

OUTPUT_DIR = "StructBERT_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nResults will be saved to: {OUTPUT_DIR}")

# ==========================================================
# 开始五折
# ==========================================================

for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):

    print("\n" + "=" * 65)
    print(f"Fold {fold + 1}/{n_splits}")
    print("=" * 65)

    val_df = df.iloc[val_idx].reset_index(drop=True)

    texts = val_df["sentence"].tolist()

    y_true = val_df["label"].tolist()

    y_pred = []

    y_prob = []

    label_map = {
        "正面": 1,
        "负面": 0
    }

    print(f"Validation Samples : {len(texts)}")

    # ======================================================
    # 批量预测
    # ======================================================

    results = semantic_cls(texts)
    # ======================================================
    # 批量预测
    # ======================================================

    for r in results:

        scores = np.array(r["scores"])

        labels = r["labels"]

        # --------------------------------------------------
        # 找到预测类别
        # --------------------------------------------------

        pred_idx = np.argmax(scores)

        pred_label_text = labels[pred_idx]

        pred_label = label_map[pred_label_text]

        y_pred.append(pred_label)

        # --------------------------------------------------
        # 提取"正类(Positive=1)"概率
        # 用于ROC-AUC
        # --------------------------------------------------

        prob_positive = 0.0

        for label, score in zip(labels, scores):

            if label == "正面":
                prob_positive = score

                break

        y_prob.append(prob_positive)

    # ======================================================
    # Accuracy
    # ======================================================

    acc = accuracy_score(
        y_true,
        y_pred
    )

    # ======================================================
    # Balanced Accuracy
    # ======================================================

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    # ======================================================
    # Precision / Recall / F1
    # ======================================================

    precision, recall, f1, _ = precision_recall_fscore_support(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    # ======================================================
    # ROC-AUC
    # ======================================================

    auc_score = roc_auc_score(
        y_true,
        y_prob
    )

    # ======================================================
    # ROC Curve
    # ======================================================

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    roc_infos.append(
        (fpr, tpr, roc_auc)
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"Fold {fold + 1} ROC")

    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"ROC_Fold_{fold + 1}.png"
        ),
        dpi=300
    )

    plt.close()

    # ======================================================
    # Confusion Matrix
    # ======================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=True
    )

    plt.title(f"Fold {fold + 1} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"ConfusionMatrix_Fold_{fold + 1}.png"
        ),
        dpi=300
    )

    plt.close()

    # ======================================================
    # 保存全部预测结果
    # ======================================================

    all_y_true.extend(y_true)

    all_y_pred.extend(y_pred)
    # ======================================================
    # Save Fold Prediction
    # ======================================================

    fold_result = pd.DataFrame({

        "True Label": y_true,

        "Pred Label": y_pred,

        "Positive Probability": y_prob

    })

    fold_result.to_excel(

        os.path.join(

            OUTPUT_DIR,

            f"Fold_{fold + 1}_Prediction.xlsx"

        ),

        index=False

    )

    print(

        f"Prediction file saved: Fold_{fold + 1}_Prediction.xlsx"

    )

    # ======================================================
    # 保存评价指标
    # ======================================================

    all_acc.append(acc)

    all_bal_acc.append(bal_acc)

    all_prec.append(precision)

    all_rec.append(recall)

    all_f1.append(f1)

    all_auc.append(auc_score)

    # ======================================================
    # 输出本折结果
    # ======================================================

    print(f"\nFold {fold + 1} Results")

    print(f"Accuracy            : {acc:.4f}")

    print(f"Balanced Accuracy   : {bal_acc:.4f}")

    print(f"Macro Precision     : {precision:.4f}")

    print(f"Macro Recall        : {recall:.4f}")

    print(f"Macro F1            : {f1:.4f}")

    print(f"ROC-AUC             : {auc_score:.4f}")
# ==========================================================
# 五折平均结果
# ==========================================================

print("\n" + "=" * 70)
print("Final 5-Fold Results")
print("=" * 70)

print(f"Mean Accuracy            : {np.mean(all_acc):.4f}")
print(f"Mean Balanced Accuracy   : {np.mean(all_bal_acc):.4f}")
print(f"Mean Macro Precision     : {np.mean(all_prec):.4f}")
print(f"Mean Macro Recall        : {np.mean(all_rec):.4f}")
print(f"Mean Macro F1            : {np.mean(all_f1):.4f}")
print(f"Mean ROC-AUC             : {np.mean(all_auc):.4f}")

print("\nStandard Deviation")

print(f"Accuracy            : {np.std(all_acc, ddof=1):.4f}")
print(f"Balanced Accuracy   : {np.std(all_bal_acc, ddof=1):.4f}")
print(f"Macro Precision     : {np.std(all_prec, ddof=1):.4f}")
print(f"Macro Recall        : {np.std(all_rec, ddof=1):.4f}")
print(f"Macro F1            : {np.std(all_f1, ddof=1):.4f}")
print(f"ROC-AUC             : {np.std(all_auc, ddof=1):.4f}")

# ==========================================================
# Mean ROC Curve
# ==========================================================

mean_fpr = np.linspace(0, 1, 100)

tprs = []

aucs = []

for fpr, tpr, roc_auc in roc_infos:
    interp_tpr = np.interp(mean_fpr, fpr, tpr)

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    aucs.append(roc_auc)

mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

plt.figure(figsize=(6, 6))

plt.plot(
    mean_fpr,
    mean_tpr,
    lw=2,
    label=f"Mean ROC (AUC={mean_auc:.4f}±{std_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    linewidth=1.2,
    color="gray"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Mean ROC Curve")

plt.legend(loc="lower right")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Mean_ROC.png"
    ),
    dpi=300
)

plt.close()

print("\nMean ROC Saved.")

# ==========================================================
# Overall Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    all_y_true,
    all_y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

fig, ax = plt.subplots(figsize=(6, 5))

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=True
)

plt.title("Overall Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Overall_ConfusionMatrix.png"
    ),
    dpi=300
)

plt.close()

print("Overall Confusion Matrix Saved.")

# ==========================================================
# 导出论文Excel
# ==========================================================

result = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Balanced Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "ROC-AUC"

    ],

    "Mean ± Std": [

        f"{np.mean(all_acc):.4f} ± {np.std(all_acc, ddof=1):.4f}",

        f"{np.mean(all_bal_acc):.4f} ± {np.std(all_bal_acc, ddof=1):.4f}",

        f"{np.mean(all_prec):.4f} ± {np.std(all_prec, ddof=1):.4f}",

        f"{np.mean(all_rec):.4f} ± {np.std(all_rec, ddof=1):.4f}",

        f"{np.mean(all_f1):.4f} ± {np.std(all_f1, ddof=1):.4f}",

        f"{np.mean(all_auc):.4f} ± {np.std(all_auc, ddof=1):.4f}"

    ]

})

result.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "StructBERT_Baseline_Results.xlsx"

    ),

    index=False

)

print("\nResults Excel Saved.")

print("\nAll Finished.")
