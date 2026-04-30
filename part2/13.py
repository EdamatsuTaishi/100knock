with open("col1.txt", "r") as fc1, open("col2.txt", "r") as fc2, open("col1-2.txt", "w") as fc12:
    for line1, line2 in zip(fc1, fc2):
        fc12.write(line1.strip() + "\t" + line2.strip() + "\n")