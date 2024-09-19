import pandas as pd


results_df = pd.read_csv('data/clean/results_filtered.csv')
races_df = pd.read_csv('data/clean/races_filtered.csv')
drivers_df = pd.read_csv('data/clean/drivers_2024.csv')
status_df = pd.read_csv('data/csv/status.csv')

race_mapping_name = races_df.set_index('raceId')['name'].to_dict()
race_mapping_year = races_df.set_index('raceId')['year'].to_dict()

driver_mapping = drivers_df.set_index('driverId')['surname'].to_dict()

status_mapping = status_df.set_index('statusId')['status'].to_dict()



results_df['raceName'] = results_df['raceId'].map(race_mapping_name)


results_df['year'] = results_df['raceId'].map(race_mapping_year)


results_df['driverName'] = results_df['driverId'].map(driver_mapping)


results_df['status'] = results_df['statusId'].map(status_mapping)

columns_to_drop = ['resultId', 'constructorId', 'number', 'raceId', 'driverId', 'statusId']
results_df = results_df.drop(columns=columns_to_drop, errors='ignore')

columns_order = ['year', 'raceName', 'driverName'] + [col for col in results_df.columns if col not in ['year', 'raceName', 'driverName']]
results_df = results_df[columns_order]

results_df['raceName'] = results_df['raceName'].fillna('Course Inconnue')
results_df['driverName'] = results_df['driverName'].fillna('Pilote Inconnu')
results_df['status'] = results_df['status'].fillna('Statut Inconnu')

results_df.to_csv('final.csv', index=False)

