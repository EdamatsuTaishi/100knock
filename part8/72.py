import torch
import pandas as pd
from gensim.models import KeyedVectors

def main():
    # データの読み込みとw2vの準備
    train = pd.read_csv("/home/edamatsu/comp/100knock/part8/SST-2/train.tsv", sep="\t")
    w2v = KeyedVectors.load_word2vec_format("/home/edamatsu/comp/100knock/part8/GoogleNews-vectors-negative300.bin.gz", binary=True)
    
    # テキストを平均ベクトルに変換する関数
    def get_mean_vector(text):
        # w2vに存在する単語のベクトルだけを抽出
        vectors = [w2v[w] for w in text.lower().split() if w in w2v.key_to_index]
        if not vectors:
            return None
        # 平均をとってtorch.Tensor化
        return torch.tensor(vectors).mean(dim=0)

    # 特徴行列 X とラベル y の作成
    # 訓練データからベクトルを取得しNone（w2vに単語がない行）を除外
    vectors = train["sentence"].apply(get_mean_vector)
    valid_idx = vectors.notna()
    
    X = torch.stack(vectors[valid_idx].tolist())
    y = torch.tensor(train.loc[valid_idx, "label"].values).float().unsqueeze(1)

    # モデルの設計（ロジスティック回帰）
    # 重みベクトルとの内積 + シグモイド = nn.Linear(dim, 1) + nn.Sigmoid()
    model = torch.nn.Sequential(
        torch.nn.Linear(w2v.vector_size, 1),
        torch.nn.Sigmoid()
    )

    # 動作確認
    with torch.no_grad():
        preds = model(X[:5])
    
    print(f"特徴量形状: {X.shape}")
    print(f"予測値（最初の5件）:\n{preds}")
    print(f"実際のラベル（最初の5件）:\n{y[:5]}")

if __name__ == "__main__":
    main()