import pandas as pd
from pandas import DataFrame
import numpy as np

train_path = "/home/edamatsu/comp/100/7/test_project/SST-2/train.tsv"
dev_path = "/home/edamatsu/comp/100/7/test_project/SST-2/dev.tsv"
test_path = "/home/edamatsu/comp/100/7/test_project/SST-2/test.tsv"


df_train = pd.read_csv(train_path, sep="\t")
df_dev = pd.read_csv(dev_path, sep="\t")
df_test = pd.read_csv(test_path, sep="\t")


df_train = df_train["label"]
df_dev = df_dev["label"]


def count_labels(df):
    pos_count = len(df[df == 1])
    neg_count = len(df[df == 0])
    return pos_count, neg_count

pos_train, neg_train = count_labels(df_train)
pos_dev, neg_dev = count_labels(df_dev) 

print(f'train_neg: {pos_train,neg_train}')
print(f'train_neg: {pos_dev,neg_dev}')