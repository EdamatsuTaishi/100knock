import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- データ準備 --- 
train_path = "/home/edamatsu/comp/100/7/test_project/SST-2/train.tsv"
dev_path   = "/home/edamatsu/comp/100/7/test_project/SST-2/dev.tsv"

df_train = pd.read_csv(train_path, sep="\t")
df_dev   = pd.read_csv(dev_path,   sep="\t")

def make_examples(df):
    examples = []
    for _, row in df.iterrows():
        tokens = row["sentence"].split()
        feats  = dict(Counter(tokens))
        examples.append({"feature": feats, "label": str(row["label"])})
    return examples

train_data = make_examples(df_train)
dev_data   = make_examples(df_dev)

vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform([d["feature"] for d in train_data])
y_train = [d["label"]         for d in train_data]
X_dev   = vec.transform([d["feature"] for d in dev_data])
y_dev   = [d["label"]         for d in dev_data]

# --- 正則化パラメータを変化させて評価 ---
Cs = [0.01, 0.1, 1, 10, 100]  # C は大きいほど正則化が弱くなる（逆強度）
accuracies = []

for C in Cs:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    accuracies.append(accuracy_score(y_dev, y_pred))

# --- グラフ描画 ---
plt.figure()
plt.plot(Cs, accuracies, marker='o')
plt.xscale('log')  # C を対数スケールで表示
plt.xlabel('Regularization parameter C (log scale)')
plt.ylabel('Accuracy on dev set')
plt.title('Dev Accuracy vs Regularization Parameter C')
plt.tight_layout()
plt.show()
plt.savefig("69.png")