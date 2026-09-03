import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# ==========================
# 1. 读取数据
# ==========================

DATA_PATH = "data.csv"
TEXT_COL = "sentence"
LABEL_COL = "label"

df = pd.read_csv(DATA_PATH)

texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].astype(int).tolist()

print(f"Total Samples: {len(texts)}")

# ==========================
# 2. 加载词典
# ==========================

def load_dict(path):

    with open(path, "r", encoding="utf-8") as f:

        return set(

            w.strip()

            for w in f

            if w.strip()

        )

lm_pos = load_dict("lm_pos.txt")
lm_neg = load_dict("lm_neg.txt")

du_pos = load_dict("du_pos.txt")
du_neg = load_dict("du_neg.txt")

print("\nDictionary Size")
print(f"LM Positive : {len(lm_pos)}")
print(f"LM Negative : {len(lm_neg)}")
print(f"DU Positive : {len(du_pos)}")
print(f"DU Negative : {len(du_neg)}")

# ==========================
# 3. 词典预测
# ==========================

def dict_predict(texts, pos_dict, neg_dict):

    predictions = []
    scores = []

    for text in tqdm(texts):

        words = jieba.lcut(text)

        pos_cnt = sum(1 for w in words if w in pos_dict)
        neg_cnt = sum(1 for w in words if w in neg_dict)

        score = pos_cnt - neg_cnt

        scores.append(score)

        if score > 0:

            predictions.append(1)

        else:

            predictions.append(0)

    return predictions, scores

# ==========================
# 4. 指标计算
# ==========================

def evaluate_macro(y_true, y_pred, scores):

    acc = accuracy_score(y_true, y_pred)

    bal_acc = balanced_accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    auc = roc_auc_score(

        y_true,

        scores

    )

    return (

        acc,

        bal_acc,

        precision,

        recall,

        f1,

        auc

    )

# ==========================
# 5. ROC曲线
# ==========================

def plot_roc(y_true, scores, save_name, title):

    fpr, tpr, _ = roc_curve(y_true, scores)

    auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6,6))

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"ROC(AUC = {auc:.4f})"

    )

    plt.plot(

        [0,1],

        [0,1],

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

    plt.savefig(save_name,dpi=300)

    plt.close()

# ==========================
# 6. 混淆矩阵
# ==========================

def plot_confusion(y_true, y_pred, save_name, title):

    cm = confusion_matrix(y_true,y_pred)

    plt.figure(figsize=(6,5))

    plt.imshow(cm,cmap="Blues")

    plt.colorbar()

    plt.xticks([0,1],["Negative","Positive"])
    plt.yticks([0,1],["Negative","Positive"])

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(

                j,

                i,

                str(cm[i,j]),

                ha="center",

                va="center"

            )

    plt.tight_layout()

    plt.savefig(save_name,dpi=300)

    plt.close()

# ==========================
# 7. 类别预测分布
# ==========================

def show_distribution(name, preds):

    s = pd.Series(preds)

    print(f"\n{name} Prediction Distribution")

    print(s.value_counts())

    print("\nPercentage")

    print(

        s.value_counts(normalize=True)

    )

# ==========================
# 8. LM词典
# ==========================

print("\n=========================")
print("LM Dictionary")
print("=========================")

lm_preds, lm_scores = dict_predict(

    texts,

    lm_pos,

    lm_neg

)

(

    lm_acc,

    lm_bal_acc,

    lm_p,

    lm_r,

    lm_f1,

    lm_auc

)=evaluate_macro(

    labels,

    lm_preds,

    lm_scores

)

show_distribution("LM",lm_preds)

plot_roc(

    labels,

    lm_scores,

    "LM_ROC.png",

    "ROC Curve"

)

plot_confusion(

    labels,

    lm_preds,

    "LM_Confusion.png",

    "Confusion Matrix"

)

# ==========================
# 9. DU词典
# ==========================

print("\n=========================")
print("DU Dictionary")
print("=========================")

du_preds, du_scores = dict_predict(

    texts,

    du_pos,

    du_neg

)

(

    du_acc,

    du_bal_acc,

    du_p,

    du_r,

    du_f1,

    du_auc

)=evaluate_macro(

    labels,

    du_preds,

    du_scores

)

show_distribution("DU",du_preds)

plot_roc(

    labels,

    du_scores,

    "DU_ROC.png",

    "ROC Curve"

)

plot_confusion(

    labels,

    du_preds,

    "DU_Confusion.png",

    "Confusion Matrix"

)

# ==========================
# 10. 保存预测结果
# ==========================

prediction_df = pd.DataFrame({

    "sentence":texts,

    "true_label":labels,

    "lm_score":lm_scores,

    "lm_prediction":lm_preds,

    "du_score":du_scores,

    "du_prediction":du_preds

})

prediction_df.to_excel(

    "dictionary_predictions.xlsx",

    index=False

)

# ==========================
# 11. 汇总结果
# ==========================

result_df = pd.DataFrame({

    "Model":[

        "LM",

        "DU"

    ],

    "Accuracy":[

        lm_acc,

        du_acc

    ],

    "Balanced Accuracy":[

        lm_bal_acc,

        du_bal_acc

    ],

    "Macro Precision":[

        lm_p,

        du_p

    ],

    "Macro Recall":[

        lm_r,

        du_r

    ],

    "Macro F1":[

        lm_f1,

        du_f1

    ],

    "ROC-AUC":[

        lm_auc,

        du_auc

    ]

})

print("\n==============================")
print(result_df)
print("==============================")

result_df.to_excel(

    "baseline_dict_macro_results.xlsx",

    index=False

)

print("\nResults Saved.")

print("1. baseline_dict_macro_results.xlsx")
print("2. dictionary_predictions.xlsx")
print("3. LM_ROC.png")
print("4. DU_ROC.png")
print("5. LM_Confusion.png")
print("6. DU_Confusion.png")