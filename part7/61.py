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