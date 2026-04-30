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

import numpy as np

# 1. 特徴量名を取得
#   scikit-learn 1.0 以降なら get_feature_names_out()
#   それ以前なら feature_names_
try:
    feature_names = vec.get_feature_names_out()
except AttributeError:
    feature_names = np.array(vec.feature_names_)

# 2. モデルの係数を取り出し
#   coef_ は形状 (1, n_features) の 2D-array
coefs = model.coef_[0]  # (n_features,)

# 3. 係数のソート用インデックスを作成
sorted_idx = np.argsort(coefs)

# 4. 下位20（最も負の重みが大きいもの）／上位20（最も正の重みが大きいもの）を取得
bottom20_idx = sorted_idx[:20]
top20_idx    = sorted_idx[-20:][::-1]  # 逆順にして最大値順に

# 5. 結果を表示
print("=== 重みが低い特徴量トップ20 （ネガティブを強く示唆） ===")
for idx in bottom20_idx:
    print(f"{feature_names[idx]:<15} : {coefs[idx]:.4f}")

print("\n=== 重みが高い特徴量トップ20 （ポジティブを強く示唆） ===")
for idx in top20_idx:
    print(f"{feature_names[idx]:<15} : {coefs[idx]:.4f}")
