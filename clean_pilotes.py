import pandas as pd

# Charger les données
drivers = pd.read_csv('data/csv/drivers.csv')
results = pd.read_csv('data/clean/results_filtered.csv')
races = pd.read_csv('data/clean/races_filtered.csv')

# Filtrer les courses de la saison 2024
races_2024 = races[races['year'] == 2024]

# Obtenir les raceId des courses de 2024
race_ids_2024 = races_2024['raceId'].unique()

# Filtrer les résultats pour les courses de 2024
results_2024 = results[results['raceId'].isin(race_ids_2024)]

# Obtenir les driverId uniques
driver_ids_2024 = results_2024['driverId'].unique()

# Filtrer les pilotes pour obtenir ceux de 2024
drivers_2024 = drivers[drivers['driverId'].isin(driver_ids_2024)]

# Sauvegarder les pilotes de 2024 dans un nouveau fichier CSV
drivers_2024.to_csv('data/clean/drivers_2024.csv', index=False)