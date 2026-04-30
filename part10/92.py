from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The movie was full of"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成設定: 確率（scores）を取得するために return_dict_in_generate と output_scores をTrueに
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=True,
        temperature=1.0,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 生成された部分のトークンID（入力分を除去）
gen_tokens = outputs.sequences[0][inputs.input_ids.shape[-1]:]
# 各ステップのロジット（scores）
gen_scores = outputs.scores

print(f"Prompt: {prompt}\n" + "-"*30)

for i, (token_id, score) in enumerate(zip(gen_tokens, gen_scores)):
    # 確率を計算 (Softmax)
    probs = torch.softmax(score[0], dim=-1)
    token_prob = probs[token_id].item()
    
    # トークンをデコード
    token_str = tokenizer.decode([token_id])
    
    print(f"Step {i+1}: '{token_str}' (Likelihood: {token_prob:.4f})")

# 最終的な文章を表示
final_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print("-" * 30 + f"\nFinal: {final_text}")