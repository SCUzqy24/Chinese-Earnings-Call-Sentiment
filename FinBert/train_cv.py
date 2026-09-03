# ============================================================
# train_cv.py
# Five-Fold Cross Validation Framework
# 以后只需要修改 MODEL_PATH 即可。
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

# ============================================================
# 自定义 Trainer
# ============================================================

from custom_trainer import CustomTrainer

# ============================================================
# 工具函数
# ============================================================

from utils import (

    compute_metrics,

    plot_roc_curve,

    plot_confusion_matrix,

    plot_mean_roc,

    save_fold_result,

    save_summary,

    print_summary,

    export_excel

)

warnings.filterwarnings("ignore")

# ============================================================
# 基础参数
# ============================================================

from modelscope import snapshot_download

MODEL_PATH = snapshot_download(
    "finbert_finetuned1"
)

print(MODEL_PATH)

DATA_PATH = "data.csv"

TEXT_COLUMN = "sentence"

LABEL_COLUMN = "label"

OUTPUT_DIR = "./output"

NUM_LABELS = 2

NUM_EPOCHS = 3

BATCH_SIZE = 16

MAX_LENGTH = 512

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

N_SPLITS = 5

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 固定随机种子
# （你的utils.py没有set_seed，因此放到主程序中）
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print(f"Random Seed = {seed}")

set_seed(RANDOM_SEED)


all_y_true = []

all_y_pred = []

all_prob = []

# ============================================================
# 创建输出目录
# ============================================================

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)

# ============================================================
# 打印实验信息
# ============================================================

print("=" * 70)
print("Five-Fold Cross Validation")
print("=" * 70)

print(f"Model          : {MODEL_PATH}")
print(f"Device         : {DEVICE}")
print(f"Epochs         : {NUM_EPOCHS}")
print(f"Batch Size     : {BATCH_SIZE}")
print(f"Learning Rate  : {LEARNING_RATE}")
print(f"Output Dir     : {OUTPUT_DIR}")

print("=" * 70)
# ============================================================
# 读取数据
# ============================================================

print("\nLoading Dataset...")

df = pd.read_csv(DATA_PATH)

# 仅保留需要的两列
df = df[[TEXT_COLUMN, LABEL_COLUMN]]

# 删除缺失值
df = df.dropna().reset_index(drop=True)

# 标签转为整数
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

print(f"Total Samples : {len(df)}")

print("\nClass Distribution")

print(df[LABEL_COLUMN].value_counts())

# ============================================================
# 加载Tokenizer
# ============================================================

print("\nLoading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(

    MODEL_PATH,

    use_fast=True

)

print("Tokenizer Loaded.")

# ============================================================
# Tokenize Function
# ============================================================

def tokenize_function(examples):
    """
    Tokenize 文本

    返回：

    input_ids
    attention_mask
    token_type_ids（部分模型自动返回）
    """

    return tokenizer(

        examples[TEXT_COLUMN],

        truncation=True,

        max_length=MAX_LENGTH,

        padding=False

    )

# ============================================================
# Dynamic Padding
# ============================================================

data_collator = DataCollatorWithPadding(

    tokenizer=tokenizer

)

# ============================================================
# 五折交叉验证
# ============================================================

skf = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=RANDOM_SEED

)

# ============================================================
# 保存所有Fold结果
# ============================================================

fold_results = []

# ============================================================
# 保存所有ROC信息
# 用于最终绘制平均ROC
# ============================================================

roc_infos = []

# ============================================================
# 打印初始化信息
# ============================================================

print("\nInitialization Finished.")

print("=" * 70)
print("Start Five-Fold Cross Validation")
print("=" * 70)
# ============================================================
# Five-Fold Cross Validation
# ============================================================

for fold, (train_idx, valid_idx) in enumerate(

        skf.split(df, df[LABEL_COLUMN]),

        start=1):

    print("\n" + "=" * 70)
    print(f"Fold {fold}/{N_SPLITS}")
    print("=" * 70)

    # ========================================================
    # 划分训练集 / 验证集
    # ========================================================

    train_df = df.iloc[train_idx].reset_index(drop=True)

    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    print("\nTrain Distribution")

    print(train_df[LABEL_COLUMN].value_counts())

    print("\nValidation Distribution")

    print(valid_df[LABEL_COLUMN].value_counts())

    # ========================================================
    # HuggingFace Dataset
    # ========================================================

    train_dataset = Dataset.from_pandas(

        train_df,

        preserve_index=False

    )

    valid_dataset = Dataset.from_pandas(

        valid_df,

        preserve_index=False

    )

    # ========================================================
    # Tokenization
    # ========================================================

    train_dataset = train_dataset.map(

        tokenize_function,

        batched=True,

        desc="Tokenizing Train"

    )

    valid_dataset = valid_dataset.map(

        tokenize_function,

        batched=True,

        desc="Tokenizing Validation"

    )

    # ========================================================
    # 删除原始文本列
    # ========================================================

    train_dataset = train_dataset.remove_columns([TEXT_COLUMN])

    valid_dataset = valid_dataset.remove_columns([TEXT_COLUMN])

    # ========================================================
    # HuggingFace Trainer要求标签列名必须叫 labels
    # ========================================================

    train_dataset = train_dataset.rename_column(

        LABEL_COLUMN,

        "labels"

    )

    valid_dataset = valid_dataset.rename_column(

        LABEL_COLUMN,

        "labels"

    )

    # ========================================================
    # Torch Tensor
    # ========================================================

    train_dataset.set_format("torch")

    valid_dataset.set_format("torch")

    # ========================================================
    # 加载模型
    # ========================================================

    print("\nLoading Model...")

    model = AutoModelForSequenceClassification.from_pretrained(

        MODEL_PATH,

        num_labels=NUM_LABELS,

        ignore_mismatched_sizes=True

    )

    model.to(DEVICE)

    print("Model Loaded.")

    # ========================================================
    # 当前Fold输出目录
    # ========================================================

    fold_output_dir = os.path.join(

        OUTPUT_DIR,

        f"fold_{fold}"

    )

    os.makedirs(

        fold_output_dir,

        exist_ok=True

    )

    # ========================================================
    # Training Arguments
    # ========================================================

    training_args = TrainingArguments(

        output_dir=fold_output_dir,

        learning_rate=LEARNING_RATE,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        weight_decay=WEIGHT_DECAY,

        logging_strategy="epoch",

        save_strategy="epoch",

        eval_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="Macro F1",

        greater_is_better=True,

        save_total_limit=1,

        seed=RANDOM_SEED,

        fp16=torch.cuda.is_available(),

        remove_unused_columns=False,

        report_to="none",
    )

    # ========================================================
    # EarlyStopping
    # ========================================================

    early_stop = EarlyStoppingCallback(

        early_stopping_patience=2,

        early_stopping_threshold=0.0

    )

    # ========================================================
    # Trainer
    # ========================================================

    print("\nInitializing Trainer...")

    trainer = CustomTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=valid_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        callbacks=[early_stop]

    )

    from torch.optim.lr_scheduler import StepLR

    trainer.create_optimizer()

    trainer.lr_scheduler = StepLR(

        trainer.optimizer,

        step_size=2,

        gamma=0.1

    )

    print("Trainer Initialized.")
# ============================================================
# 开始训练
# ============================================================

    print("\nStart Training...\n")

    trainer.train()

    print("\nTraining Finished.")

    # ========================================================
    # 保存最佳模型
    # ========================================================

    trainer.save_model(fold_output_dir)

    tokenizer.save_pretrained(fold_output_dir)

    print(f"Best model saved to: {fold_output_dir}")

    # ========================================================
    # 验证集预测
    # ========================================================

    print("\nPredicting Validation Set...")

    predictions = trainer.predict(valid_dataset)

    logits = predictions.predictions

    probabilities = torch.softmax(

        torch.tensor(logits, dtype=torch.float32),

        dim=1

    ).cpu().numpy()

    y_pred = np.argmax(

        probabilities,

        axis=1

    )

    y_true = valid_df[LABEL_COLUMN].values

    print("Prediction Finished.")

    # ========================================================
    # 六项评价指标
    # ========================================================

    acc = accuracy_score(

        y_true,

        y_pred

    )

    bal_acc = balanced_accuracy_score(

        y_true,

        y_pred

    )

    precision, recall, f1, _ = precision_recall_fscore_support(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    roc_auc = roc_auc_score(

        y_true,

        probabilities[:, 1]

    )

    # ========================================================
    # ROC Curve
    # ========================================================

    roc_info = plot_roc_curve(

        y_true=y_true,

        y_prob=probabilities[:, 1],

        save_path=os.path.join(

            fold_output_dir,

            "ROC.png"

        ),

        title=f"Fold {fold} ROC Curve"

    )

    roc_infos.append(roc_info)

    # ========================================================
    # Confusion Matrix
    # ========================================================

    plot_confusion_matrix(

        y_true=y_true,

        y_pred=y_pred,

        save_path=os.path.join(

            fold_output_dir,

            "ConfusionMatrix.png"

        )

    )
    all_y_true.extend(y_true)

    all_y_pred.extend(y_pred)

    # ========================================================
    # 保存预测结果
    # ========================================================

    prediction_df = pd.DataFrame({

        "true_label": y_true,

        "pred_label": y_pred,

        "prob_negative": probabilities[:, 0],

        "prob_positive": probabilities[:, 1]

    })

    prediction_df.to_csv(

        os.path.join(

            fold_output_dir,

            "validation_prediction.csv"

        ),

        index=False,

        encoding="utf-8-sig"

    )

    # ========================================================
    # 输出当前Fold结果
    # ========================================================

    print("\n" + "=" * 60)

    print(f"Fold {fold} Results")

    print("=" * 60)

    print(f"Accuracy            : {acc:.4f}")

    print(f"Balanced Accuracy   : {bal_acc:.4f}")

    print(f"Macro Precision     : {precision:.4f}")

    print(f"Macro Recall        : {recall:.4f}")

    print(f"Macro F1            : {f1:.4f}")

    print(f"ROC-AUC             : {roc_auc:.4f}")

    # ========================================================
    # 保存当前Fold结果
    # ========================================================

    fold_results.append({

        "Fold": fold,

        "Accuracy": acc,

        "Balanced Accuracy": bal_acc,

        "Macro Precision": precision,

        "Macro Recall": recall,

        "Macro F1": f1,

        "ROC-AUC": roc_auc

    })
# ============================================================
# Five-Fold Finished
# ============================================================

print("\n")
print("=" * 70)
print("Five-Fold Cross Validation Finished")
print("=" * 70)
# ==========================================================
# Overall Confusion Matrix
# ==========================================================

ocm = confusion_matrix(
    all_y_true,
    all_y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=ocm,
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

# ============================================================
# DataFrame
# ============================================================

results_df = pd.DataFrame(fold_results)

# ============================================================
# 保存五折结果(csv)
# ============================================================

save_fold_result(

    metrics=fold_results,

    save_path=os.path.join(

        OUTPUT_DIR,

        "cv_results.csv"

    )

)

# ============================================================
# 保存五折Excel
# ============================================================

results_df.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "cv_results.xlsx"

    ),

    engine="openpyxl",

    index=False

)

# ============================================================
# 保存Mean±Std(csv)
# ============================================================

save_summary(

    metrics=fold_results,

    save_path=os.path.join(

        OUTPUT_DIR,

        "summary.csv"

    )

)

# ============================================================
# 保存论文Excel
# ============================================================

export_excel(

    metrics=fold_results,

    save_path=os.path.join(

        OUTPUT_DIR,

        "summary.xlsx"

    )

)

# ============================================================
# 打印最终结果
# ============================================================

print_summary(

    fold_results

)

# ============================================================
# 绘制五折平均ROC
# ============================================================

plot_mean_roc(

    roc_infos,

    os.path.join(

        OUTPUT_DIR,

        "Mean_ROC.png"

    )

)

# ============================================================
# 保存Mean、Std（便于论文复制）
# ============================================================

summary_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Balanced Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "ROC-AUC"

    ],

    "Mean": [

        results_df["Accuracy"].mean(),

        results_df["Balanced Accuracy"].mean(),

        results_df["Macro Precision"].mean(),

        results_df["Macro Recall"].mean(),

        results_df["Macro F1"].mean(),

        results_df["ROC-AUC"].mean()

    ],

    "Std": [

        results_df["Accuracy"].std(ddof=1),

        results_df["Balanced Accuracy"].std(ddof=1),

        results_df["Macro Precision"].std(ddof=1),

        results_df["Macro Recall"].std(ddof=1),

        results_df["Macro F1"].std(ddof=1),

        results_df["ROC-AUC"].std(ddof=1)

    ]

})

summary_df.to_excel(

    os.path.join(

        OUTPUT_DIR,

        "summary_detail.xlsx"

    ),

    engine="openpyxl",

    index=False

)

print("\n")
print("=" * 70)
print("All Experiments Finished Successfully!")
print("=" * 70)

print(f"\nResults Saved In : {OUTPUT_DIR}")