#45
import os
from google import genai

# APIキーを手動で設定
GEMINI_API_KEY = "Google_key"  # ここに実際のAPIキーを入力
client = genai.Client(api_key=GEMINI_API_KEY)


chat = client.chats.create(model = 'gemini-1.5-flash')


# 質問内容
question = ("つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。")

response = chat.send_message(question)

print("1回目 : " , response.text)



# 質問内容
questions = ("さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？")

response = chat.send_message(questions)



print("2回目 : " , response.text)



