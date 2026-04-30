from collections import Counter
from itertools import islice

with open("popular-names.txt", "r") as f:
    lines = f.readlines()
    
col1_vals = [line.split()[0] for line in lines]
freq = Counter(col1_vals)

for word, count in freq.most_common():
    print(f"{count:>5} {word}")