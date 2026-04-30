import torch
import pandas as pd
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, TensorDataset

def main():
    # データとWord2Vecの準備
    train_df = pd.read_csv("/home/edamatsu/comp/100knock/part8/SST-2/train.tsv", sep="\t")
    w2v = KeyedVectors.load_word2vec_format("/home/edamatsu/comp/100knock/part8/GoogleNews-vectors-negative300.bin.gz", binary=True)

    # テキストを平均ベクトルに変換（w2vにある単語のみ）
    def get_features(df):
        vecs, labels = [], []
        for _, row in df.iterrows():
            words = [w2v[w] for w in row["sentence"].lower().split() if w in w2v.key_to_index]
            if words:
                vecs.append(torch.tensor(words).mean(dim=0))
                labels.append([float(row["label"])])
        return torch.stack(vecs), torch.tensor(labels)

    X_train, y_train = get_features(train_df)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # モデル・損失関数・最適化の定義
    # 学習対象はモデルのLinear層（重み）のみで，w2vは上記で変換済みのため固定された状態
    model = torch.nn.Sequential(torch.nn.Linear(w2v.vector_size, 1), torch.nn.Sigmoid())
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 学習ループ
    print(f"Starting training on {len(X_train)} samples...")
    for epoch in range(10):
        total_loss = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(loader):.4f}")

    # 保存
    torch.save(model.state_dict(), "model/model.pth")

if __name__ == "__main__":
    main()