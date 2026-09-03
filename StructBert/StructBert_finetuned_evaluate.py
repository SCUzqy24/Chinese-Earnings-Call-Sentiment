# ==========================================================
# Import Libraries
# ==========================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Metrics
from modelscope.hub.api import HubApi
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    precision_recall_fscore_support,

    roc_auc_score,

    roc_curve,

    auc,

    confusion_matrix,

    ConfusionMatrixDisplay

)

warnings.filterwarnings("ignore")

# ==========================================================
# Basic Parameters
# ==========================================================

MODEL_ID = "structbert_finetuned1"

DATA_PATH = "data.csv"

TEXT_COLUMN = "sentence"

LABEL_COLUMN = "label"

MAX_EPOCHS = 3

N_SPLITS = 5

RANDOM_SEED = 42

OUTPUT_DIR = "StructBERT_finetuned2_Results"

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)

# ==========================================================
# Login ModelScope
# ==========================================================

api = HubApi()

api.login("ms-1cf30d6b-f58a-4c71-a2a2-280c3faad891")

# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading Dataset...")

df = pd.read_csv(DATA_PATH)

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

print(f"Total Samples : {len(df)}")

print("\nClass Distribution")

print(df[LABEL_COLUMN].value_counts())

# ==========================================================
# Stratified Five-Fold
# ==========================================================

skf = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=RANDOM_SEED

)

# ==========================================================
# Containers for Metrics
# ==========================================================

all_acc = []

all_bal_acc = []

all_prec = []

all_rec = []

all_f1 = []

all_auc = []


# ==========================================================
# Containers for Whole Dataset Prediction
# ==========================================================

all_y_true = []

all_y_pred = []

all_prob = []

prediction_records = []

roc_infos = []

print("\nInitialization Finished.")

print("=" * 70)

print("Start Five-Fold Cross Validation")

print("=" * 70)

# ==========================================================
# Start Five-Fold Cross Validation
# ==========================================================

for fold, (train_idx, val_idx) in enumerate(

        skf.split(df, df[LABEL_COLUMN])

):

    print("\n" + "=" * 70)

    print(f"Fold {fold + 1}/{N_SPLITS}")

    print("=" * 70)

    # ======================================================
    # Split Dataset
    # ======================================================

    train_df = df.iloc[train_idx].reset_index(drop=True)

    valid_df = df.iloc[val_idx].reset_index(drop=True)

    print("\nTrain Distribution")

    print(train_df[LABEL_COLUMN].value_counts())

    print("\nValidation Distribution")

    print(valid_df[LABEL_COLUMN].value_counts())

    # ======================================================
    # Save CSV
    # ======================================================

    train_file = f"train_fold_{fold + 1}.csv"

    valid_file = f"valid_fold_{fold + 1}.csv"

    train_df.to_csv(

        train_file,

        index=False

    )

    valid_df.to_csv(

        valid_file,

        index=False

    )

    # ======================================================
    # Load MsDataset
    # ======================================================

    train_dataset = MsDataset.load(

        "csv",

        data_files=train_file

    )

    valid_dataset = MsDataset.load(

        "csv",

        data_files=valid_file

    )

    WORK_DIR = f"workspace_fold_{fold + 1}"


    # ======================================================
    # Modify Config
    # （保持与你原来的7010wa完全一致）
    # ======================================================

    def cfg_modify_fn(cfg):

        cfg.train.max_epochs = MAX_EPOCHS

        cfg.train.hooks = [

            {

                "type": "TextLoggerHook",

                "interval": 100

            },

            {

                "type": "CheckpointHook",

                "interval": 1

            }

        ]

        cfg.evaluation.metrics = [

            Metrics.seq_cls_metric

        ]

        cfg["dataset"] = {

            "train": {

                "labels": [

                    "0",

                    "1"

                ],

                "first_sequence": TEXT_COLUMN,

                "label": LABEL_COLUMN,

            }

        }

        cfg.train.lr_scheduler = {

            "type": "StepLR",

            "step_size": 2,

            "options": {

                "warmup": {

                    "type": "LinearWarmup",

                    "warmup_iters": 1

                }

            }

        }

        return cfg


    # ======================================================
    # Build Trainer
    # ======================================================

    kwargs = dict(

        model=MODEL_ID,

        train_dataset=train_dataset,

        eval_dataset=valid_dataset,

        work_dir=WORK_DIR,

        cfg_modify_fn=cfg_modify_fn

    )

    trainer = build_trainer(

        name="nlp-base-trainer",

        default_args=kwargs

    )

    print("\nStart Training...")

    trainer.train()

    print("Training Finished.")

    # ======================================================
    # Load Fine-tuned Model
    # ======================================================

    semantic_cls = pipeline(

        Tasks.text_classification,

        model=f"{WORK_DIR}/output"

    )

    print("Fine-tuned model loaded.")

    # ======================================================
    # Validation Prediction
    # ======================================================

    texts = valid_df[TEXT_COLUMN].tolist()

    y_true = valid_df[LABEL_COLUMN].tolist()

    y_pred = []

    y_prob = []

    results = semantic_cls(texts)

    label_map = {
        "1": 1,
        "0": 0
    }

    # ======================================================
    # Prediction
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

            if label == "1":
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
        colorbar=False
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

        "StructBERT_finetuned2_Results.xlsx"

    ),

    index=False

)

print("\nResults Excel Saved.")

print("\nAll Finished.")