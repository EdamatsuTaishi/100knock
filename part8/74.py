import torch
import pandas as pd
from gensim.models import KeyedVectors

def main():
    # モデルの構造とWord2Vec,開発データの読み込み
    w2v_path = "/home/edamatsu/comp/100knock/part8/GoogleNews-vectors-negative300.bin.gz"
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    dev_df = pd.read_csv("/home/edamatsu/comp/100knock/part8/SST-2/dev.tsv", sep="\t")

    # モデルの定義（学習済み重みをロードする場合を想定）
    model = torch.nn.Sequential(torch.nn.Linear(w2v.vector_size, 1), torch.nn.Sigmoid())
    model.load_state_dict(torch.load("model/model.pth"))

    # 開発セットを特徴ベクトル(X)とラベル(y)に変換
    def get_features(df):
        vecs, labels = [], []
        for _, row in df.iterrows():
            words = [w2v[w] for w in row["sentence"].lower().split() if w in w2v.key_to_index]
            if words:
                vecs.append(torch.tensor(words).mean(dim=0))
                labels.append([float(row["label"])])
        return torch.stack(vecs), torch.tensor(labels)

    X_dev, y_dev = get_features(dev_df)

    # 評価（推論モード）
    model.eval()
    with torch.no_grad():
        outputs = model(X_dev)
        # 0.5より大きければポジティブ(1) 小さければネガティブ(0)と判定
        preds = (outputs > 0.5).float()
        
        # 正解数をカウントして正解率を算出
        accuracy = (preds == y_dev).sum().item() / len(y_dev)

    print(f"開発セットのサンプル数: {len(y_dev)}")
    print(f"開発セットの正解率: {accuracy:.4f}")

if __name__ == "__main__":
    main()