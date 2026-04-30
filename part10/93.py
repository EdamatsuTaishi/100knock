import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

sentences = [
    "The movie was full of surprises",    # 正解
    "The movies were full of surprises",   # 正解
    "The movie were full of surprises",    # 誤（単複不一致）
    "The movies was full of surprises",    # 誤（単複不一致）
]

def get_ppl(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        # labelsにinput_idsを渡すと、内部で自動的にシフトしてLossを計算してくれる
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    
    return math.exp(loss)

print(f"{'Sentence':<40} | {'PPL':>7}")
print("-" * 50)
for s in sentences:
    ppl = get_ppl(s)
    print(f"{s:<40} | {ppl:>7.2f}")