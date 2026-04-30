def ntlines(N, file):
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines[-N:]:
            print(line.strip())

ntlines(4, "popular-names.txt")