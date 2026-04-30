from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The movie was full of incomprehensibilities."

tokens = tokenizer.tokenize(text)

print(f"元の文: {text}")
for i in range(len(tokens)):
    print(f"トークン {i}: {tokens[i]}")