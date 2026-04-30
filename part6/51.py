from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('/home/edamatsu/comp/100-66/test_project/GoogleNews-vectors-negative300.bin', binary=True)


print(wv.similarity('United_States', 'U.S.'))