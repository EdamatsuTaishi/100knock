from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルとトークナイザーの準備（軽量なGPT-2を使用）
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# プロンプトのトークン化と確認
prompt = "The movie was full of"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"トークンID: {inputs['input_ids'][0].tolist()}")
print(f"トークン列: {[tokenizer.decode([t]) for t in inputs['input_ids'][0]]}")

# 予測と確率算出
with torch.no_grad():
    logits = model(**inputs).logits[0, -1, :] # 最後のトークンの出力のみ取得
    probs = torch.softmax(logits, dim=-1)

# 上位10個の表示
top_k = torch.topk(probs, k=10)
for i, (idx, p) in enumerate(zip(top_k.indices, top_k.values)):
    print(f"{i+1}: {tokenizer.decode([idx])} ({p:.4f})")