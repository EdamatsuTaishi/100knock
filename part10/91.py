from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The movie was full of"
inputs = tokenizer(prompt, return_tensors="pt")

# 検証したい設定をリスト化（引数名をgenerateの仕様に合わせる）
configs = [
    {"label": "Greedy (Standard)", "do_sample": False},
    {"label": "Greedy (Low Temp)", "do_sample": False, "temperature": 0.5}, # Greedyに温調は本来無意味ですが比較用
    {"label": "Beam Search", "num_beams": 5, "do_sample": False},
    {"label": "Top-K Sampling", "do_sample": True, "top_k": 50, "temperature": 0.7},
    {"label": "Top-P (Nucleus) Sampling", "do_sample": True, "top_p": 0.9, "temperature": 1.2},
]

for config in configs:
    # 表示用のラベルを取り出して残りを引数として渡す
    label = config.pop("label")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=15, 
            pad_token_id=tokenizer.eos_token_id,
            **config  # 辞書の内容を引数として展開
        )
    
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[{label}]\n -> {res}\n")