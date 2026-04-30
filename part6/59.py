import gensim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. モデルと国名リストの読み込み
model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/edamatsu/comp/100/6/test_project/GoogleNews-vectors-negative300.bin', binary=True
)
with open('/home/edamatsu/comp/100/6/test_project/countries.txt', 'r') as f:
    countries = [c.replace(' ', '_') for c in f.read().splitlines()]

# 2. モデルに含まれる国名のみ残す
countries = [c for c in countries if c in model.key_to_index]

# 3. ベクトル取得＆NumPy 配列化
vectors = np.array([model[c] for c in countries])

# 4. t-SNE による次元削減（max_iter に変更）
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,      # 旧 n_iter→max_iter
    random_state=42
)
vectors_2d = tsne.fit_transform(vectors)

# 5. 可視化
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=30)

for (x, y), label in zip(vectors_2d, countries):
    plt.text(x + 0.5, y + 0.5, label, fontsize=9)

plt.title('t-SNE visualization of country word vectors')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()

plt.savefig('sample_1.png')