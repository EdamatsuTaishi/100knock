import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import os
import evaluate

# GPUの設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0のみ使用
torch.cuda.set_device(0)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# num_labels=2 (Negative/Positive) を指定
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# データの読み込みとDataset変換
train_df = pd.read_csv("ch08/SST-2/train.tsv", sep="\t", header=0)
dev_df = pd.read_csv("ch08/SST-2/dev.tsv", sep="\t", header=0)
train_dataset = Dataset.from_pandas(train_df[['sentence', 'label']])
dev_dataset = Dataset.from_pandas(dev_df[['sentence', 'label']])

# 前処理
def preprocess_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# 評価指標
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 学習の設定
training_args = TrainingArguments(
    output_dir="ch09/results/87",
    learning_rate=1e-5,           # ファインチューニング用の小さな学習率
    per_device_train_batch_size=32,
    per_device_eval_batch_size=256,
    num_train_epochs=10,
    weight_decay=0.01,
    fp16=True,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    save_strategy="epoch",
    save_total_limit=1,
    metric_for_best_model="accuracy",
    logging_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
)

# トレーナーの実行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# 学習
trainer.train()

# 検証セットで最終評価
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")