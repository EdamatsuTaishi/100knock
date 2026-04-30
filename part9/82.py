import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "The movie was full of [MASK]."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = logits[0, mask_token_index, :]
# softmaxを適用して確率を計算
probabilities = torch.nn.functional.softmax(mask_token_logits, dim=1)

top_k = 10
# 上位top_kのトークンとその確率を取得
top_values, top_indices = torch.topk(probabilities, top_k, dim=1)

print(f"Input: {text}\n")
print(f"{'Rank':<5} | {'Token':<15} | {'Probability':<10}")
print("-" * 35)

for i in range(top_k):
    token = tokenizer.decode([top_indices[0][i]])
    prob = top_values[0][i].item()
    print(f"{i+1:<5} | {token:<15} | {prob:.4f}")