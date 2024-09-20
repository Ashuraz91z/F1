import pandas as pd

drivers = pd.read_csv('data/csv/drivers.csv')
results = pd.read_csv('data/clean/results_filtered.csv')
races = pd.read_csv('data/clean/races_filtered.csv')

races_2024 = races[races['year'] == 2024]

race_ids_2024 = races_2024['raceId'].unique()

results_2024 = results[results['raceId'].isin(race_ids_2024)]

driver_ids_2024 = results_2024['driverId'].unique()

drivers_2024 = drivers[drivers['driverId'].isin(driver_ids_2024)]

drivers_2024.to_csv('data/clean/drivers_2024.csv', index=False)