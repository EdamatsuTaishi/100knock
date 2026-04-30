unique_vals = set()
with open("popular-names.txt", "r") as f:
    for line in f:
        unique_vals.add(line.split()[0])

print("\n".join(sorted(unique_vals)))