import pandas as pd
from pathlib import Path

# 1. Charger les données
results_df = pd.read_csv('final.csv')

# 2. Nettoyer les données (facultatif mais recommandé)
# Par exemple, remplacer les valeurs '\N' par NaN ou une valeur appropriée
results_df.replace('\\N', pd.NA, inplace=True)

# 3. Gérer les caractères spéciaux dans les noms de dossiers
def sanitize_folder_name(name):
    # Remplacer les caractères non valides par des underscores
    return "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name]).strip()

# 4. Définir le répertoire de sortie
output_base_dir = Path('Resultat Triés')

# Créer le répertoire de base s'il n'existe pas
output_base_dir.mkdir(parents=True, exist_ok=True)

# 5. Grouper les données par raceName et year
grouped = results_df.groupby(['raceName', 'year'])

# 6. Itérer sur chaque groupe et sauvegarder les résultats
for (race_name, year), group in grouped:
    # Sanitize race_name pour être utilisé comme nom de dossier
    sanitized_race_name = sanitize_folder_name(race_name)
    
    # Définir le chemin du dossier pour ce race et année
    race_dir = output_base_dir / sanitized_race_name / str(year)
    
    # Créer les dossiers nécessaires
    race_dir.mkdir(parents=True, exist_ok=True)
    
    # Définir le chemin du fichier CSV
    output_csv = race_dir / 'results.csv'
    
    # Sauvegarder le groupe dans le fichier CSV
    group.to_csv(output_csv, index=False)
    
    print(f"Sauvegardé : {output_csv}")

print("Tous les fichiers ont été organisés avec succès.")