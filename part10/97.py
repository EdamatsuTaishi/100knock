import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
import evaluate
import os

# GPUの設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0のみ使用
torch.cuda.set_device(0)

# モデル定義
class EmbeddingClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, freeze_encoder=True):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # 分類層（Feedforward）
        self.classifier = nn.Linear(hidden_size, num_labels)

        # encoderを固定（課題の意図）
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLSベクトル取得
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# 評価指標
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():

    model_name = "llm-jp/llm-jp-3-150m-instruct3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EmbeddingClassifier(model_name)

    # データ読み込み
    train_df = pd.read_csv("ch07/SST-2/train.tsv", sep="\t")
    dev_df = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t")

    # Dataset化
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(dev_df)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )

    # Trainer設定
    training_args = TrainingArguments(
        output_dir="./ch10/97/results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir="./ch10/97/logs",
        learning_rate=5e-4,   # classifierのみ学習なので少し大きめ
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 学習
    trainer.train()

    # 評価
    results = trainer.evaluate()
    print("最終評価:", results)


if __name__ == "__main__":
    main()