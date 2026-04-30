from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import Counter

train_path = "/home/edamatsu/comp/100/7/test_project/SST-2/train.tsv"
dev_path = "/home/edamatsu/comp/100/7/test_project/SST-2/dev.tsv"
test_path = "/home/edamatsu/comp/100/7/test_project/SST-2/test.tsv"

# pandas.DataFrameで読み込み
df_train = pd.read_csv(train_path, sep="\t")
df_dev = pd.read_csv(dev_path, sep="\t")
df_test = pd.read_csv(test_path, sep="\t")


def make_examples(df: pd.DataFrame) -> list[dict]:
    examples = []
    for _, row in df.iterrows():
        text  = row["sentence"]
        label = str(row["label"])             # '0' や '1' の文字列に
        tokens = text.split()                 # スペース区切りでトークン化
        feat   = dict(Counter(tokens))        # 単語ごとの出現頻度を辞書化
        examples.append({
            "text":    text,
            "label":   label,
            "feature": feat
        })
    return examples

train_data = make_examples(df_train)
dev_data   = make_examples(df_dev)

# 学習データの最初の事例を表示して目視で確認
print(train_data[0])
# 1. DictVectorizer で BoW 辞書を数値ベクトルに変換
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform([ex["feature"] for ex in train_data])
y_train = [ex["label"] for ex in train_data]

X_dev = vec.transform([ex["feature"] for ex in dev_data])
y_dev = [ex["label"] for ex in dev_data]

# 2. ロジスティック回帰モデルの定義・学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from collections import Counter

def predict_sentiment(text: str, 
                      vec: DictVectorizer, 
                      model: LogisticRegression) -> None:
    """
    テキストのポジティブ(1)/ネガティブ(0)を予測し、
    各クラスの確率も表示する。
    """
    # 1. トークン化＆BoW 辞書化
    tokens  = text.split()
    feature = dict(Counter(tokens))
    
    # 2. ベクトル化
    X = vec.transform([feature])   # shape = (1, vocab_size)
    
    # 3. 予測
    pred   = model.predict(X)[0]
    probs  = model.predict_proba(X)[0]
    labels = model.classes_        # ['0','1']
    
    # 4. 結果表示
    label_name = "ネガティブ" if pred == '0' else "ポジティブ"
    print(f"入力テキスト: 「{text}」")
    print(f"→ 予測ラベル: {pred} ({label_name})")
    print("→ 各クラスの確率:")
    for cls, p in zip(labels, probs):
        name = "ネガティブ" if cls=='0' else "ポジティブ"
        print(f"   P(y={cls}／{name}) = {p:.4f}")

# ————— 実行例 —————
example = "the worst movie I 've ever seen"
predict_sentiment(example, vec, model)
