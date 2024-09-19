import pandas as pd

# Chemin des fichiers
path_races_filtered = "data/clean/races_filtered.csv"
path_results = "data/csv/results.csv"
output_path = "data/clean/results_filtered.csv"

# Charger les raceId à partir de races_filtered.csv
df_races_filtered = pd.read_csv(path_races_filtered, usecols=["raceId"])

# Charger les données du fichier results.csv
df_results = pd.read_csv(path_results)

# Filtrer les résultats pour inclure uniquement ceux dont le raceId est présent dans races_filtered.csv
df_results_filtered = df_results[df_results['raceId'].isin(df_races_filtered['raceId'])]

# Sauvegarder les résultats filtrés dans un nouveau fichier CSV
df_results_filtered.to_csv(output_path, index=False)

