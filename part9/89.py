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

# 事前学習済みモデルの読み込み
model_name = "doyoungkim/bert-base-uncased-finetuned-sst2"
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
    output_dir="ch09/results/89",
    learning_rate=1e-5,           # ファインチューニング用の小さな学習率
    per_device_train_batch_size=32,
    per_device_eval_batch_size=256,
    num_train_epochs=3,
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
# 学習前の評価
print("Evaluating BEFORE training")
pre_eval_results = trainer.evaluate()
print(f"Pre-training Validation Accuracy: {pre_eval_results['eval_accuracy']:.4f}")
print("-" * 30)

# 学習
trainer.train()

# 検証セットで最終評価
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

trainer.save_model("ch09/results/89/final_model")
tokenizer.save_pretrained("ch09/results/89/final_model")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 学習済みモデルのパス
model_path = "ch09/results/89/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

test_sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

print(f"Using device: {device}")
print("-" * 30)

for text in test_sentences:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    prob, pred_idx = torch.max(probs, dim=-1)

    label = "Positive" if pred_idx.item() == 1 else "Negative"

    print(f"Text: {text}")
    print(f"Prediction: {label} ({prob.item():.2%})\n")