#46
import os
from google import genai

# APIキーを手動で設定
GEMINI_API_KEY = "Google_key"  # ここに実際のAPIキーを入力
client = genai.Client(api_key=GEMINI_API_KEY)

# 質問内容
question = ("五・七・五（十七音）のリズムで詠む川柳というものが日本にはあります。川柳の案を10個作成せよ、このとき次のルールを守れ、川柳のルールは、主に次のとおりです。五・七・五（十七音）のリズムで詠む、上五・中七・下五の三つの部分から成る（三句体）、現代仮名遣い、口語体で表記する﻿俳句のような切れ字や季語は不要、字余りや字足らずがあっても、全体のバランスが崩れない限り定型感は保つことができる")

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=question,
)

print(response.text)





