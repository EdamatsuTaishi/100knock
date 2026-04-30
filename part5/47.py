#47

import os
from google import genai

lists = []

# APIキーを手動で設定
GEMINI_API_KEY = "Google_key"  # ここに実際のAPIキーを入力
client = genai.Client(api_key=GEMINI_API_KEY)


haiku = ["今日も一日　頑張る自分が　好きだよな", "猫が寝る顔　平和な時間　癒されるね", "朝のコーヒー　香りに包まれ　幸せかな", "満員電車で　隣の人と　無言の挨拶","夕焼け空に　染まる雲見て　明日への力","冷たいビール　喉を潤す　至福の時間", "週末の朝は　ゆっくりと寝て　充電完了"
,"好きな音楽　心踊る　最高の時間", "新しい靴で　街を歩く　気分爽快" , "本を読む時間　静かな空間　心が安らぐ"]


# 質問内容
question = ("各川柳の面白さを1から10の10段階でそれぞれ評価せよ、この時それぞれの川柳に数字を1つのみ出力せよ　：　")

for i in range(len(haiku)):
  response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=question + haiku[i],
    )
  print(response.text)
  lists.append(int(response.text))



