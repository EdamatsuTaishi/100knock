import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)


# GPU設定

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# データ読み込み


train_path = "./ch07/SST-2/train.tsv"
dev_path = "./ch07/SST-2/dev.tsv"

train_df = pd.read_csv(train_path, sep="\t")
dev_df = pd.read_csv(dev_path, sep="\t")

train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)


# トークナイザ


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2はpad_tokenが無い


# プロンプト作成


def create_prompt(example):
    label_text = "positive" if example["label"] == 1 else "negative"

    text = f"""You are a sentiment classifier.

Sentence: {example["sentence"]}
Sentiment: {label_text}"""

    return {"text": text}

train_dataset = train_dataset.map(create_prompt)
dev_dataset = dev_dataset.map(create_prompt)


# トークナイズ


def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # labels を追加
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)

# 不要列削除
train_dataset = train_dataset.remove_columns(["sentence", "label", "text"])
dev_dataset = dev_dataset.remove_columns(["sentence", "label", "text"])

train_dataset.set_format(type="torch")
dev_dataset.set_format(type="torch")


# モデル


model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)


# 学習設定


training_args = TrainingArguments(
    output_dir="./ch10/results/gpt2-sst2",
    per_device_train_batch_size=8,   # OOM回避で8推奨
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none"   # wandb無効
)


# Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

trainer.train()


# 保存
save_dir = "./ch10/results/gpt2-sst2"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 推論テスト
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_dir = "./ch10/results/gpt2-sst2"

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

prompt = """You are a sentiment classifier.

Sentence: I love this movie.
Sentiment:"""

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=3,
    do_sample=False
)

print(tokenizer.decode(output[0]))