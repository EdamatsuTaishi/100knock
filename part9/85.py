import pandas as pd
from transformers import AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# SST-2のTSVにはヘッダーがあるためにheader=0 を指定する
train_df = pd.read_csv('/home/edamatsu/comp/100knock/part8/SST-2/train.tsv', sep='\t', header=0)
dev_df = pd.read_csv('/home/edamatsu/comp/100knock/part8/SST-2/dev.tsv', sep='\t', header=0)

print("Train Data Preview")
print(train_df.head())

# テキストをトークン列に変換する関数
def tokenize_text(text):
    # 文をトークン（文字列）のリストに変換(cls_tokenやsep_tokenは含まれない)
    return tokenizer.tokenize(text)

# 全データに適用
# 新しいカラム 'tokens' にリスト形式で保存します
train_df['tokens'] = train_df['sentence'].apply(tokenize_text)
dev_df['tokens'] = dev_df['sentence'].apply(tokenize_text)

print("\nTokenization Result (Example 1)")
print(f"Original: {train_df['sentence'][0]}")
print(f"Label: {train_df['label'][0]}")
print(f"Tokens: {train_df['tokens'][0]}")