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


########################################################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_metrics(y_true, y_pred, prefix=""):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='1')
    rec = recall_score(y_true, y_pred, pos_label='1')
    f1 = f1_score(y_true, y_pred, pos_label='1')
    print(f"{prefix}Accuracy : {acc:.4f}")
    print(f"{prefix}Precision: {prec:.4f}")
    print(f"{prefix}Recall   : {rec:.4f}")
    print(f"{prefix}F1-score : {f1:.4f}")
    print("-" * 30)

# 学習データ上の予測
y_pred_train = model.predict(X_train)
print_metrics(y_train, y_pred_train, prefix="Train set ")

# 検証データ上の予測
y_pred_dev = model.predict(X_dev)
print_metrics(y_dev, y_pred_dev, prefix="Dev   set ")
