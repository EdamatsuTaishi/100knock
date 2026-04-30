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

# 全トークンの平均ベクトルを取得する関数
def get_mean_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state は (バッチサイズ, トークン数, ベクトルの次元数)
        # 文のトークン方向の次元で平均を取る
        mean_embedding = torch.mean(outputs.last_hidden_state[0], dim=0)
    return mean_embedding

# 全文のベクトルを計算
embeddings = [get_mean_embedding(s) for s in sentences]

print(f"{'Sentence A':<35} | {'Sentence B':<35} | {'Similarity'}")
print("-" * 85)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        vec_i = embeddings[i].reshape(1, -1)
        vec_j = embeddings[j].reshape(1, -1)
        sim = cosine_similarity(vec_i, vec_j)[0][0]
        
        print(f"{sentences[i]:<35} | {sentences[j]:<35} | {sim:.4f}")