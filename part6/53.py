from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('/home/edamatsu/comp/100/6/test_project/GoogleNews-vectors-negative300.bin', binary=True)
results = wv.most_similar(positive=['Spain','Athens'], negative=['Madrid'], topn=10)

for result in results:
    print(result)