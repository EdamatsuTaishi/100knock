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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 検証データで予測を行い、混同行列を計算
y_pred = model.predict(X_dev)
labels = model.classes_  # 例: ['0','1']
cm = confusion_matrix(y_dev, y_pred, labels=labels)

# プロットの準備
fig, ax = plt.subplots()
im = ax.imshow(cm, aspect='equal')

# 軸目盛りにラベルを付与
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)

# 軸ラベルとタイトル
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix')

# 各セル内に数値を注釈
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center')

# カラーバー
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
plt.savefig("66.png")