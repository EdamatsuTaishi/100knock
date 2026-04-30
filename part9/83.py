import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

# 文をベクトル（埋め込み）に変換する関数
def get_cls_embedding(text):
    # clsトークンの埋め込みを取得
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # 最終層の[CLS]トークン（0番目のトークン）を取り出す
        cls_embedding = outputs.last_hidden_state[0, 0, :]
    return cls_embedding

# 全文のベクトルを取得
embeddings = [get_cls_embedding(s) for s in sentences]

# 全ての組み合わせでコサイン類似度を計算
print(f"{'Sentence A':<35} | {'Sentence B':<35} | {'Similarity'}")
print("-" * 85)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        # 2次元配列に変換して類似度計算
        vec_i = embeddings[i].reshape(1, -1)
        vec_j = embeddings[j].reshape(1, -1)
        sim = cosine_similarity(vec_i, vec_j)[0][0]
        
        print(f"{sentences[i]:<35} | {sentences[j]:<35} | {sim:.4f}")