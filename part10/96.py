import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm



# プロンプト作成(few-shot)
def create_prompt(text):
    return f"""You are a sentiment classifier.

Classify the sentiment of the following sentence.
Answer with only one word: positive or negative.
Do not explain.

Sentence: I love this movie.
Sentiment: positive

Sentence: This was terrible.
Sentiment: negative

Sentence: {text}
Sentiment:"""



# 予測関数

def predict_sentiment(text, model, tokenizer):
    prompt = create_prompt(text)

    # トークナイズ & GPUへ
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,      # 1語だけで十分
            do_sample=False,       # 決定論的に
            temperature=0.0,
        )

    # 入力部分を除いて生成部分のみ取得
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response = response.lower().strip()

    if "positive" in response:
        return 1
    elif "negative" in response:
        return 0
    else:
        return 0  # fallback


def main():
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   # GPU高速化
        device_map="auto"            # 自動でGPU割り当て
    )

    model.eval()
    dev_path = "ch07/SST-2/dev.tsv"
    dataset = pd.read_csv(dev_path, sep="\t")

    correct = 0
    total = 0

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text = row["sentence"]
        label = row["label"]  # 0: negative, 1: positive

        predicted_label = predict_sentiment(text, model, tokenizer)

        if predicted_label == label:
            correct += 1

        total += 1

    accuracy = correct / total
    print("\n---RESULT---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct : {correct}")
    print(f"Total   : {total}")


if __name__ == "__main__":
    main()