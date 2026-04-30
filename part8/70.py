import numpy as np
from gensim.models import KeyedVectors
from typing import Dict, Tuple

def load_word_embeddings(
    model_path: str, 
    limit: int = None
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    
    # モデルの読み込み
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=limit)
    
    vocab_size = len(model.index_to_key)
    embedding_dim = model.vector_size

    # 単語埋め込み行列の作成 (V+1, D)
    # np.zerosで初期化するため，0行目は自動的にゼロベクトル(PAD用)へ
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    
    # 2行目以降に事前学習済みベクトルをコピー
    embedding_matrix[1:] = model.vectors

    # 単語とIDの双方向対応付け
    word_to_id = {"<PAD>": 0}
    for i, word in enumerate(model.index_to_key, start=1):
        word_to_id[word] = i
        
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return embedding_matrix, word_to_id, id_to_word

def main():
    FILE_PATH = "/home/edamatsu/comp/100knock/part8/GoogleNews-vectors-negative300.bin.gz"
    
    # limitはメモリ消費を抑えるために書いてるだけで，とりあえず全件(None)で実行
    matrix, w2i, i2w = load_word_embeddings(FILE_PATH, limit=None)

    # 要件の確認
    print(f"行列の形状 (V+1, D): {matrix.shape}")
    print(f"ID 0 の単語: {i2w[0]}")
    print(f"ID 0 のベクトル（先頭5要素）: {matrix[0][:5]}")
    print(f"ID 1 の単語: {i2w[1]}")
    print(f"ID 1 のベクトル（先頭5要素）: {matrix[1][:5]}")

if __name__ == "__main__":
    main()