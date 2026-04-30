import torch
import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import DPOTrainer, DPOConfig
import os

# GPU設定

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# データ読み込み


train_path = "./ch07/SST-2/train.tsv"
train_df = pd.read_csv(train_path, sep="\t")


# DPO用データ作成


def build_dpo_sample(row):
    sentence = row["sentence"]
    label = row["label"]

    prompt = f"""You are a sentiment classifier.

Sentence: {sentence}
Sentiment:"""

    if label == 1:
        chosen = " positive"
        rejected = " negative"
    else:
        chosen = " negative"
        rejected = " positive"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

dpo_data = [build_dpo_sample(row) for _, row in train_df.iterrows()]
dataset = Dataset.from_list(dpo_data)


# tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# モデル


model = GPT2LMHeadModel.from_pretrained("gpt2")

# 参照モデル（DPOで必要）
ref_model = GPT2LMHeadModel.from_pretrained("gpt2")


# 学習設定


training_args = DPOConfig(
    output_dir="./ch10/results/gpt2-dpo",
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)


# Trainer


trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()


# 保存
model.save_pretrained("./ch10/results/gpt2-dpo")
tokenizer.save_pretrained("./ch10/results/gpt2-dpo")

print("DPO training finished.")

# 推論
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_dir = "./ch10/results/gpt2-dpo"

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

prompt = """You are a sentiment classifier.

Sentence: This movie was amazing.
Sentiment:"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=3,
    do_sample=False
)

print(tokenizer.decode(outputs[0]))