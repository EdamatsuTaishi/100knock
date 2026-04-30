with open("popular-names.txt", "r") as f:
    lines = f.readlines()

print("".join(sorted(lines, key=lambda x: float(x.split()[2]), reverse=True)))