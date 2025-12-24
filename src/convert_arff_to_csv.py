import pandas as pd

arff_file = "data/chronic_kidney_disease.arff"
csv_file = "data/ckd.csv"

columns = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane", "classification"
]

rows = []
data_started = False

with open(arff_file, "r") as file:
    for line in file:
        line = line.strip()

        if not line or line.startswith("%"):
            continue

        if line.lower() == "@data":
            data_started = True
            continue

        if data_started:
            # Clean row
            line = line.replace("?", "")
            values = [v.strip() for v in line.split(",")]

            # FIX: enforce exact column length
            if len(values) > len(columns):
                values = values[:len(columns)]
            elif len(values) < len(columns):
                values.extend([""] * (len(columns) - len(values)))

            rows.append(values)

df = pd.DataFrame(rows, columns=columns)
df.to_csv(csv_file, index=False)

print("SUCCESS: ckd.csv created with cleaned columns")
