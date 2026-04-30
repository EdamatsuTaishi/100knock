import json

with open('jawiki-country.json', 'r') as f:
    ctx = [json.loads(line) for line in f]

entry = next(item for item in ctx if item['title'] == 'イギリス')

import re

uk = entry['text']
pattern = r"\[\[ファイル:([^|\]]+)"
print(re.findall(pattern, uk))

# 5. re.findall()を使って、正規表現にマッチする部分を全て抽出する