def nhlines(N, file):
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines[:N]:
            print(line.strip())

nhlines(4, "popular-names.txt")