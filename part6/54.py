import pandas as pd
from gensim.models import KeyedVectors

with open('/home/edamatsu/comp/100/6/test_project/questions-words.txt', 'r') as f:
    lines = f.readlines()


dataset = []
category = None

for line in lines:
    if line.startswith(':'):
       category = line[2:-2]
     
    
    else:
        words = [category] + line.split() 
        dataset.append(words)

import gensim
from tqdm import tqdm

model = gensim.models.KeyedVectors.load_word2vec_format('/home/edamatsu/comp/100/6/test_project/GoogleNews-vectors-negative300.bin', binary=True)

for i, lst in enumerate(tqdm(dataset)):
    pred, prob = model.most_similar(positive = lst[2:4], negative = lst[1:2], topn = 1)[0]
    dataset[i].append(pred)

dataset.to_csv('dataset.csv', index=False)

print(pd.DataFrame(dataset[:10]))