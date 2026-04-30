from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "bert-base-uncased"
# WordPieceトークナイザー
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "The movie was full of [MASK]."
# PyTorchテンソルに変換
inputs = tokenizer(text, return_tensors="pt")
# 勾配計算無しでシーケンス双方向で予測(logitsは各トークンの語彙に対する"スコア")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits
# [MASK]トークンの位置を特定し，その位置の予測を取得
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = predictions[0, mask_token_index, :]

# 最も高いスコアを持つトークンを予測
predicted_index = torch.argmax(mask_token_logits, dim=1).item()
predicted_token = tokenizer.decode([predicted_index])

print(f"元の文: {text}")
print(f"予測されたトークン: {predicted_token}")