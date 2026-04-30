def nsplit(N, input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    split_size = total_lines // N
    rem = total_lines % N
    
    start = 0
    base, ext = input_file.rsplit('.', 1)
    
    for i in range(N):
        end = start + split_size + (1 if i < rem else 0)
        output_file = f"{base}-{i:02}.{ext}"
        with open(output_file, "w") as out_file:
            out_file.write("".join(lines[start:end]))
        start = end

nsplit(3, "popular-names.txt")