import re
import json

with open('jawiki-country.json', 'r') as f:
    ctx = [json.loads(line) for line in f]

entry = next(item for item in ctx if item['title'] == 'イギリス')

uk = entry['text']
pattern = r"(==+)\s*(.+?)\s*\1"
sections = re.findall(pattern, uk)

for level, name in sections:
    print(f"{name}: {len(level) - 1}")

# 4. re.findall()を使って、正規表現にマッチする部分を全て抽出する