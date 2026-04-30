import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# GPUの設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0のみ使用
torch.cuda.set_device(0)


model_id = "Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if torch.cuda.is_available():
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map = "auto"
else:
    dtype = torch.float32
    device_map = None

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device_map,
)

chat = [
    {"role": "system", "content": "You are a helpful assistant. Please answer the following questions."},
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
    {"role": "assistant", "content": "A sweet eaten after dinner is called a dessert."},
]


text = tokenizer.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated, skip_special_tokens=True)

print("生成された応答:")
print(response)