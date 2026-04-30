import json

with open('jawiki-country.json', 'r') as f:
    ctx = [json.loads(line) for line in f]

entry = next(item for item in ctx if item['title'] == 'イギリス')

print(entry['text'])

# 1. json.loads()を使って、json形式の文字列をPythonの辞書型に変換する