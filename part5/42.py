import csv
import time
import requests
import io
from google import genai

# APIキーの設定
GEMINI_API_KEY = "Google_key"
client = genai.Client(api_key=GEMINI_API_KEY)

# CSVファイルのURL
csv_url = "https://raw.githubusercontent.com/nlp-waseda/JMMLU/refs/heads/main/JMMLU/world_history.csv"

import pandas as pd
import io

res = requests.get(csv_url).content
df = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None, index_col=None)
cnt = 0

import re

for i in range(df.shape[0]):
    questions = (df.iloc[i,0] + "この問いに次に示すAからDで答えてください。解答はA,B,C,Dのいずれかのみで答えてください。理由や説明は不要です。",
            "A: "+ df.iloc[i,1],
            "B: "+df.iloc[i,2],
            "C: "+df.iloc[i,3],
            "D: "+df.iloc[i,4],
            )

    response = client.models.generate_content(
      model="gemini-1.5-flash",
      contents=questions,
      )

    print("問題 : " , questions)
    print("解答 : " + response.text)

    extracted = re.findall(r'\b[A-D]\b', response.text.strip().upper())
    answer = extracted[0] if extracted else ""  # 最初に出現したA〜Dを使う


    if(df.iloc[i,5] == answer):
      cnt+=1

    print("本来の正解 : " + df.iloc[i,5])
    print("正解数 : " , cnt )
    time.sleep(4)


print(f"正解率: {cnt/df.shape[0] * 100:.2f}% （全 {df.shape[0]} 問中 {cnt} 問正解）")