import gensim
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 1. モデルと国名リストの読み込み
model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/edamatsu/comp/100/6/test_project/GoogleNews-vectors-negative300.bin',
    binary=True
)

with open('/home/edamatsu/comp/100/6/test_project/countries.txt', 'r') as f:
    countries = f.read().splitlines()

# 空白をアンダースコアに置換し，モデルにある国名のみ抽出
countries = [c.replace(' ', '_') for c in countries]
countries = [c for c in countries if c in model.key_to_index]

# 2. ベクトル化
country_vectors = [model[c] for c in countries]

# 3. 階層的クラスタリング（Ward 法）によるリンク行列の生成
#    linkage の method='ward' が Ward 法を指定
Z = linkage(country_vectors, method='ward')

# 4. デンドログラムの描画
plt.figure(figsize=(10, 6))
dendrogram(
    Z,
    labels=countries,
    leaf_rotation=90,    # ラベルを縦に回転
    leaf_font_size=8     # フォントサイズ調整
)
plt.title('Ward 法による階層クラスタリングのデンドログラム')
plt.ylabel('距離')
plt.tight_layout()
plt.show()


plt.savefig('sample.png')