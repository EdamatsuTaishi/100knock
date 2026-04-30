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

# 3. 学習完了の確認と検証データでの簡易評価
print("学習完了")
acc = model.score(X_dev, y_dev)
print(f"検証データにおける正解率: {acc:.4f}")

feat_first = dev_data[0]["feature"]
X_first = vec.transform([feat_first])
y_actual = dev_data[0]["label"]

y_pred = model.predict(X_first)[0]

print(f"予測ラベル: {y_pred}")
print(f"実際のラベル: {y_actual}")
print("一致しています。" if y_pred == y_actual else "一致していません。")

print("y_pred:", y_pred)
