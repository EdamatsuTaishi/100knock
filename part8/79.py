import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ準備
def prepare_data(file_path, word_to_id):
    df = pd.read_csv(file_path, sep="\t")
    processed = []
    for _, row in df.iterrows():
        ids = [word_to_id[w] for w in row["sentence"].lower().split() if w in word_to_id]
        if ids:
            processed.append({"input_ids": torch.tensor(ids), "label": torch.tensor([float(row["label"])])})
    return processed

def collate(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    return {
        "input_ids": pad_sequence([item["input_ids"] for item in batch], batch_first=True).to(device),
        "label": torch.stack([item["label"] for item in batch]).to(device)
    }

# アーキテクチャ: Diffusion-inspired Text Classifier
class DiffusionTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=300):
        super().__init__()
        # ファインチューニング可能な埋め込み層
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        
        # 拡散プロセス（ノイズ付加）の強さ
        self.noise_std = 0.1
        
        # Denoising Block (ResNet風の多層構造)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 埋め込み取得 [batch, seq, dim]
        x = self.embedding(x)
        
        # 拡散的ノイズ付加 (学習時のみ連続空間にノイズを混ぜる)
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        # 平均ベクトル化 (文章全体の特徴)
        x = torch.mean(x, dim=1)
        
        # ノイズ除去・特徴洗練 (Residual connection)
        x = x + self.denoiser(x)
        
        # 分類
        return self.classifier(x)

# 学習・評価ループ
def train_and_eval(model, loader, criterion, optimizer=None, desc=""):
    model.train() if optimizer else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in pbar:
            outputs = model(batch["input_ids"])
            loss = criterion(outputs, batch["label"])
            
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            correct += ((outputs > 0.5) == batch["label"]).sum().item()
            total += len(batch["label"])
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")
            
    return total_loss / len(loader), 100 * correct / total

def test_model(model, word_to_id, test_sentences):
    model.eval()
    print("\n学習済みモデルの分類テスト")
    with torch.no_grad():
        for sentence in test_sentences:
            # 前処理（訓練時と同じ）
            words = sentence.lower().split()
            ids = [word_to_id.get(w, 0) for w in words if w in word_to_id]
            if not ids:
                continue
                
            input_ids = torch.tensor([ids]).to(device)
            pred = model(input_ids).item()
            
            result = "ポジティブ" if pred > 0.5 else "ネガティブ"
            confidence = abs(pred - 0.5) * 200
            print(f"'{sentence}' → {result} (確信度: {confidence:.1f}%)")

def main():
    # パス設定
    MODEL_PATH = "/home/edamatsu/comp/100knock/part8/GoogleNews-vectors-negative300.bin.gz"
    TRAIN_TSV = "/home/edamatsu/comp/100knock/part8/SST-2/train.tsv"
    DEV_TSV = "/home/edamatsu/comp/100knock/part8/SST-2/dev.tsv"

    # 語彙構築
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    dev_df = pd.read_csv(DEV_TSV, sep="\t")
    vocab = set(" ".join(train_df["sentence"].str.lower()).split()) | set(" ".join(dev_df["sentence"].str.lower()).split())
    
    # 埋め込みロード
    wv = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
    word_to_id = {"<PAD>": 0}; embeddings = [torch.zeros(wv.vector_size)]
    for word in vocab:
        if word in wv:
            word_to_id[word] = len(word_to_id); embeddings.append(torch.tensor(wv[word]))
    emb_matrix = torch.stack(embeddings)

    # データローダー (Batch Sizeを上げると高速化)
    train_loader = DataLoader(prepare_data(TRAIN_TSV, word_to_id), batch_size=64, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(prepare_data(DEV_TSV, word_to_id), batch_size=64, collate_fn=collate)

    model = DiffusionTextClassifier(emb_matrix).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    for epoch in range(10):
        t_loss, t_acc = train_and_eval(model, train_loader, criterion, optimizer, f"Epoch {epoch+1} [Train]")
        d_loss, d_acc = train_and_eval(model, dev_loader, criterion, desc=f"Epoch {epoch+1} [Dev]")
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.2f}%, Dev Acc {d_acc:.2f}%")
# test
    test_sentences = [
        "This movie was absolutely fantastic!",  # 明確ポジ
        "The worst film I've ever seen.",       # 明確ネガ
        "It's okay, nothing special.",          # 中間
        "Brilliant acting and stunning visuals.", # ポジ
        "Boring plot and poor direction.",      # ネガ
    "A decent film with some good moments." # 中間
    ]

    test_model(model, word_to_id, test_sentences)

if __name__ == "__main__":
    main()