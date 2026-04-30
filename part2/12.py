with open("popular-names.txt", "r") as f, open("col1.txt", "w") as fc1, open("col2.txt", "w") as fc2:
    for line in f:
        columns = line.split("\t")
        fc1.write(columns[0] + "\n")
        fc2.write(columns[1] + "\n") 