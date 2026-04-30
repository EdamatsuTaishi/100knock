with open("popular-names.txt", "r") as f:
    lines = (line.replace("\t", " ") for line in f)
    for i, line in enumerate(lines):
        print(line, end="") 