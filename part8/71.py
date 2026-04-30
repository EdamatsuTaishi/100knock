import pandas as pd
import torch
from gensim.models import KeyedVectors

def main():
    # データの読み込み
    train = pd.read_csv("/home/edamatsu/comp/100knock/part8/SST-2/train.tsv", sep="\t")
    dev = pd.read_csv("/home/edamatsu/comp/100knock/part8/SST-2/dev.tsv", sep="\t")
    
    # Word2Vecの読み込み
    w2v_path = "/home/edamatsu/comp/100knock/part8/GoogleNews-vectors-negative300.bin.gz"
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
    # 語彙辞書の作成（積集合 & を使い、w2vに存在する単語のみ抽出）
    # 全テキストの単語集合を取得
    vocab_in_data = set(" ".join(train["sentence"]).lower().split()) | set(" ".join(dev["sentence"]).lower().split())
    # w2vのキーとの共通部分をID化
    valid_words = vocab_in_data & set(w2v.key_to_index)
    word_to_id = {w: i + 1 for i, w in enumerate(sorted(valid_words))}
    word_to_id["<PAD>"] = 0

    # 変換関数の定義
    def encode(text):
        ids = [word_to_id[w] for w in text.lower().split() if w in word_to_id]
        return torch.tensor(ids) if ids else None

    # データの処理
    def process(df):
        # 変換して、空（None）になった行を捨てる
        df["input_ids"] = df["sentence"].apply(encode)
        df = df.dropna(subset=["input_ids"])
        # 必要なカラムだけを辞書形式のリストにする
        return df[["sentence", "label", "input_ids"]].to_dict(orient="records")

    train_data = process(train)
    dev_data = process(dev)

    # 確認
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")
    if train_data:
        print(f"Sample Text: {train_data[0]['sentence']}")
        print(f"Sample Label: {train_data[0]['label']}")
        print(f"Sample IDs: {train_data[0]['input_ids']}")

if __name__ == "__main__":
    main()