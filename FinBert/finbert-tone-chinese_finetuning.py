import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

############################################################
# 参数
############################################################
from modelscope import snapshot_download

MODEL_PATH = snapshot_download(
    "finbert-tone-chinese"
)

TRAIN_FILE = "train.csv"

DEV_FILE = "dev.csv"

OUTPUT_DIR = "./output_fb_fintuned1"

TEXT_COLUMN = "sentence"

LABEL_COLUMN = "label"

MAX_LENGTH = 512

NUM_LABELS = 2

BATCH_SIZE = 16

NUM_EPOCHS = 8

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################################
# 固定随机种子
############################################################

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

set_seed()

############################################################
# 数据
############################################################

train_df = pd.read_csv(TRAIN_FILE)

dev_df = pd.read_csv(DEV_FILE)

train_dataset = Dataset.from_pandas(train_df)

dev_dataset = Dataset.from_pandas(dev_df)

############################################################
# tokenizer
############################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize(batch):

    return tokenizer(

        batch[TEXT_COLUMN],

        truncation=True,

        max_length=MAX_LENGTH

    )

train_dataset = train_dataset.map(tokenize, batched=True)

dev_dataset = dev_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns([TEXT_COLUMN])

dev_dataset = dev_dataset.remove_columns([TEXT_COLUMN])

train_dataset = train_dataset.rename_column(LABEL_COLUMN,"labels")

dev_dataset = dev_dataset.rename_column(LABEL_COLUMN,"labels")

train_dataset.set_format("torch")

dev_dataset.set_format("torch")

############################################################
# Data Collator
############################################################

data_collator = DataCollatorWithPadding(tokenizer)

############################################################
# model
############################################################

model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_PATH,

    num_labels=NUM_LABELS,

    ignore_mismatched_sizes=True

)

model.to(DEVICE)

############################################################
# TrainingArguments
############################################################

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    weight_decay=WEIGHT_DECAY,

    logging_strategy="epoch",

    save_strategy="epoch",

    eval_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    save_total_limit=1,

    fp16=torch.cuda.is_available(),

    report_to="none",

    lr_scheduler_type="cosine",

    warmup_ratio=0.1


)

############################################################
# Trainer
############################################################

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=dev_dataset,

    processing_class=tokenizer,

    data_collator=data_collator

)

############################################################
# Train
############################################################

print("="*60)

print("Start Training")

print("="*60)

trainer.train()

############################################################
# Save
############################################################

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print()

print("="*60)

print("Model Saved!")

print(OUTPUT_DIR)

print("="*60)