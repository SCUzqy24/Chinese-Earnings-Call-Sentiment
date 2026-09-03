"""
==========================================================
utils.py

公共工具函数

包括：

1. 固定随机种子
2. HuggingFace Trainer评价指标
3. ROC曲线
4. Confusion Matrix
5. 保存实验结果
6. 导出论文Excel

Author:
==========================================================
"""

import os
import random

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
# ==========================================================
# HuggingFace Trainer评价指标
# ==========================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Trainer.evaluate() 使用

    返回：

    Accuracy
    Balanced Accuracy
    Macro Precision
    Macro Recall
    Macro F1
    ROC-AUC（二分类）

    Parameters
    ----------
    eval_pred

    Returns
    -------
    Dict
    """

    logits, labels = eval_pred

    probabilities = torch.softmax(
        torch.tensor(logits, dtype=torch.float32),
        dim=1
    ).cpu().numpy()

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics = {

        "Accuracy": accuracy,

        "Balanced Accuracy": balanced_acc,

        "Macro Precision": precision,

        "Macro Recall": recall,

        "Macro F1": f1,

    }

    # ----------------------------
    # 二分类时计算ROC-AUC
    # ----------------------------
    if probabilities.shape[1] == 2:

        try:
            metrics["ROC-AUC"] = roc_auc_score(
                labels,
                probabilities[:, 1]
            )
        except ValueError:
            metrics["ROC-AUC"] = np.nan

    return metrics
# ==========================================================
# 绘制ROC曲线
# ==========================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    title: str = "ROC Curve"
):
    """
    绘制ROC曲线，并返回ROC信息。

    Parameters
    ----------
    y_true : ndarray
        真实标签

    y_prob : ndarray
        正类预测概率（shape=(N,)）

    save_path : str
        图片保存路径

    title : str
        图片标题

    Returns
    -------
    tuple
        (fpr, tpr, auc_value)
    """

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
        linewidth=2.0,
        label=f"AUC = {auc_value:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="gray"
    )

    plt.xlim([0, 1])
    plt.ylim([0, 1.05])

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

    print(f"ROC Curve Saved -> {save_path}")

    return fpr, tpr, auc_value
# ==========================================================
# 绘制Confusion Matrix
# ==========================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    labels=("Negative", "Positive")
):
    """
    绘制混淆矩阵。

    Parameters
    ----------
    y_true

    y_pred

    save_path

    labels
        类别名称
    """

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

    threshold = np.max(cm) / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12
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
# 保存每折评价指标
# ==========================================================

def save_fold_result(
    metrics: List[Dict],
    save_path: str
):
    """
    保存五折每一折评价指标。

    Parameters
    ----------
    metrics : List[Dict]
        每折评价指标列表

    save_path : str
        csv保存路径
    """

    df = pd.DataFrame(metrics)

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Fold Results Saved -> {save_path}")
# ==========================================================
# 保存Mean±Std
# ==========================================================

def save_summary(
    metrics: List[Dict],
    save_path: str
):
    """
    保存Mean±Std统计结果。

    Parameters
    ----------
    metrics : List[Dict]

    save_path : str
    """

    df = pd.DataFrame(metrics)

    summary = pd.DataFrame({

        "Metric": df.columns,

        "Mean": df.mean().values,

        "Std": df.std(ddof=1).values

    })

    summary.to_csv(

        save_path,

        index=False,

        encoding="utf-8-sig"

    )

    print(f"Summary Saved -> {save_path}")
# ==========================================================
# 打印最终结果
# ==========================================================

def print_summary(
    metrics: List[Dict]
):
    """
    打印五折平均结果。
    """

    df = pd.DataFrame(metrics)

    print("\n")
    print("=" * 65)
    print("Final 5-Fold Cross Validation Results")
    print("=" * 65)

    for column in df.columns:

        mean = df[column].mean()

        std = df[column].std(ddof=1)

        print(
            f"{column:<22}"
            f"{mean:.4f} ± {std:.4f}"
        )

    print("=" * 65)
# ==========================================================
# 导出论文Excel
# ==========================================================

def export_excel(
    metrics: List[Dict],
    save_path: str
):
    """
    导出论文结果Excel。

    Parameters
    ----------
    metrics

    save_path
    """

    df = pd.DataFrame(metrics)

    result = pd.DataFrame({

        "Metric": df.columns,

        "Mean ± Std": [

            f"{df[col].mean():.4f} ± {df[col].std(ddof=1):.4f}"

            for col in df.columns

        ]

    })

    result.to_excel(

        save_path,
        engine="openpyxl",
        index=False

    )

    print(f"Excel Saved -> {save_path}")
# ==========================================================
# 五折平均ROC
# ==========================================================

def plot_mean_roc(
    roc_infos,
    save_path
):
    """
    绘制五折平均ROC。

    Parameters
    ----------
    roc_infos :
        [(fpr,tpr,auc), ...]

    save_path :
        图片保存路径
    """

    mean_fpr = np.linspace(
        0,
        1,
        100
    )

    tprs = []

    aucs = []

    for fpr, tpr, auc_value in roc_infos:

        interp = np.interp(
            mean_fpr,
            fpr,
            tpr
        )

        interp[0] = 0.0

        tprs.append(interp)

        aucs.append(auc_value)

    mean_tpr = np.mean(
        tprs,
        axis=0
    )

    mean_tpr[-1] = 1.0

    mean_auc = auc(
        mean_fpr,
        mean_tpr
    )

    std_auc = np.std(
        aucs,
        ddof=0
    )

    plt.figure(figsize=(6, 6))

    plt.plot(

        mean_fpr,

        mean_tpr,

        linewidth=2,

        label=f"Mean ROC (AUC={mean_auc:.4f}±{std_auc:.4f})"

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

    plt.title("Mean ROC Curve")

    plt.legend(loc="lower right")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(f"Mean ROC Saved -> {save_path}")