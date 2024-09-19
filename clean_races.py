import pandas as pd

csv_path = "data/clean/circuits_filtered.csv"

df = pd.read_csv(csv_path, usecols=["circuitId"])

id_races = []
for value in df['circuitId']:
    id_races.append(value)

path_races = "data/csv/races.csv"

df_races = pd.read_csv(path_races)

df_races_filtered = df_races[(df_races['circuitId'].isin(id_races)) & (df_races['year'] >= 2020)]
df_races_filtered.to_csv("data/clean/races_filtered.csv", index=False)
