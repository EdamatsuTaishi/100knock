import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors

# デバイスの自動選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # データをデバイスへ転送
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True).to(device)
    labels = torch.stack([item["label"] for item in batch]).to(device)
    return {"input_ids": input_ids, "label": labels}

class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embedding_matrix, mode="mean", freeze=True)
        self.linear = nn.Linear(embedding_matrix.size(1), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(self.embedding(x)))

def train_and_eval(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            # データはcollate関数ですでにdeviceへ移動済み
            outputs = model(batch["input_ids"])
            loss = criterion(outputs, batch["label"])
            
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            correct += ((outputs > 0.5) == batch["label"]).sum().item()
            total += len(batch["label"])
            
    return total_loss / len(loader), 100 * correct / total

def main():
    train_df = pd.read_csv("/home/edamatsu/comp/100knock/part8/SST-2/train.tsv", sep="\t")
    dev_df = pd.read_csv("/home/edamatsu/comp/100knock/part8/SST-2/dev.tsv", sep="\t")
    vocab = set(" ".join(train_df["sentence"].str.lower()).split()) | set(" ".join(dev_df["sentence"].str.lower()).split())
    
    wv = KeyedVectors.load_word2vec_format("/home/edamatsu/comp/100knock/part8/GoogleNews-vectors-negative300.bin.gz", binary=True)
    word_to_id = {"<PAD>": 0}; embeddings = [torch.zeros(wv.vector_size)]
    
    for word in vocab:
        if word in wv:
            word_to_id[word] = len(word_to_id)
            embeddings.append(torch.tensor(wv[word]))
    emb_matrix = torch.stack(embeddings)

    train_loader = DataLoader(prepare_data("/home/edamatsu/comp/100knock/part8/SST-2/train.tsv", word_to_id), batch_size=8, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(prepare_data("/home/edamatsu/comp/100knock/part8/SST-2/dev.tsv", word_to_id), batch_size=8, collate_fn=collate)

    # モデルをデバイスに移動
    model = MeanEmbeddingClassifier(emb_matrix).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        t_loss, t_acc = train_and_eval(model, train_loader, criterion, optimizer)
        d_loss, d_acc = train_and_eval(model, dev_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.2f}%, Dev Acc {d_acc:.2f}%")

if __name__ == "__main__":
    main()