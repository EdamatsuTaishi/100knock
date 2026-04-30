import torch
from transformers import AutoTokenizer
import pandas as pd

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# SST-2の冒頭4事例を模したもの
sentences = [
    "hide new secretions from the parental units",
    "contains no wit , only labored gags",
    "that loves its characters and communicates something rather beautiful",
    "a computer-generated marginally animated movie"
]
labels = [0, 0, 1, 0] # 0: negative, 1: positive

# ミニバッチの作成
# padding=True で最長文に合わせて長さを揃える
# truncation=True でモデルの最大長（512）を超えた分を切り捨て
inputs = tokenizer(
    sentences, 
    padding=True, 
    truncation=True, 
    return_tensors="pt"  # PyTorchのテンソル形式で返す
)

print("Mini-batch Input IDs")
print(inputs["input_ids"])

print("\nAttention Mask")
print(inputs["attention_mask"])

# ラベルもテンソルに変換
label_tensor = torch.tensor(labels)
print("\nLabels")
print(label_tensor)