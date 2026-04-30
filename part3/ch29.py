import json

with open('jawiki-country.json', 'r') as f:
    ctx = [json.loads(line) for line in f]

entry = next(item for item in ctx if item['title'] == 'イギリス')

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

import re

uk = entry['text']
pattern = r"\|\s*(.*?)\s*=\s*(.*?)\n"
emph_pattern = r"'{2,5}"
link_pattern = r"\[\[(.+\||)(.+?)\]\]"
ref_pattern = r"\<ref.*?\>.*?\<\/ref\>|\<ref.*?\>"
br_pattern = r"\<br\s*\/?\>"
file_pattern = r"\[\[ファイル:(.*?)\|.*\]\]"
lang_pattern = r".*\{\{lang\|.*\|(.*?)\}\}"
olink_pattern = r"\{\{.*?\|(.*?)\|.*\}\}"

info_dict = {key.strip(): value.strip() for key, value in template("基礎情報", uk)}


import urllib

url = 'https://www.mediawiki.org/w/api.php?action=query&titles=File:' + urllib.parse.quote(info_dict['国旗画像']) + '&format=json&prop=imageinfo&iiprop=url'
connection = urllib.request.urlopen(urllib.request.Request(url))
response = json.loads(connection.read().decode())
iurl = response['query']['pages']['-1']['imageinfo'][0]['url']
print(iurl)

