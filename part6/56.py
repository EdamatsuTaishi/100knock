import pandas as pd
import gensim

df = pd.read_csv('/home/edamatsu/comp/100/6/test_project/wordsim353/combined.csv',header=None,names = ['単語1','単語2','人間','ベクトル'])[1:]
model = gensim.models.KeyedVectors.load_word2vec_format('/home/edamatsu/comp/100/6/test_project/GoogleNews-vectors-negative300.bin', binary=True)

sim = []
for index, i in df.iterrows():
    sim.append(model.similarity(i["単語1"], i["単語2"]))


df['ベクトル'] = sim

from scipy.stats import spearmanr
spearman_corr, _ = spearmanr(df['人間'], df['ベクトル'])
print('Spearman correlation:', spearman_corr)