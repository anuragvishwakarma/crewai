# debug_csv.py
with open("/Users/koki/Documents/testing/Rag/data/synthetic_maintenance_records.csv", "r") as f:
    for i, line in enumerate(f, 1):
        if len(line.strip().split(";")) != 7:
            print(f"Line {i}: {line.strip()}")