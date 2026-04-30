import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 学習済みモデルのパス
model_path = "ch09/results/87/checkpoint-8420"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

test_sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

print(f"Using device: {device}")
print("-" * 30)

# 推論
for text in test_sentences:
    # 入力の準備
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # 確率に変換
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # 最大値のインデックスと確率を取得
    prob, pred_idx = torch.max(probs, dim=-1)
    
    label = "Positive" if pred_idx.item() == 1 else "Negative"
    
    print(f"Text: {text}")
    print(f"Prediction: {label} ({prob.item():.2%})\n")