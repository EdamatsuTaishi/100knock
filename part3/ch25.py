import json

with open('jawiki-country.json', 'r') as f:
    ctx = [json.loads(line) for line in f]

entry = next(item for item in ctx if item['title'] == 'イギリス')


import re

uk = entry['text']

def template(name, text):
    temp_pattern = r"\{\{" + name + r"(.*)\}\}"
    pattern = r"\|\s*(.*?)\s*=\s*(.*?)\n"
    tmp = re.findall(temp_pattern, text, flags=re.DOTALL)
    
    cnt = 2
    ctx = ""
    for char in tmp[0]:
        ctx += char
        if char == "{":
            cnt += 1
        elif char == "}":
            cnt -= 1
        if cnt == 0:
            ctx = re.sub(r'^.*\n', '', ctx[:-2])
            break
        
    return re.findall(pattern, ctx)

results = template("基礎情報", uk)
info_dict = {key.strip(): value.strip() for key, value in results}

for key, value in info_dict.items():
    print(f"{key}: {value}")

# 6. re.findall()を使って、正規表現にマッチする部分を全て抽出する   