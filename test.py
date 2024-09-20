import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Charger les données avec '\\N' comme valeurs manquantes
df = pd.read_csv('final.csv', na_values=['\\N'])

# 2. Filtrer les pilotes inconnus et les données incomplètes
df = df[df['driverName'] != 'Pilote Inconnu']
df = df.dropna(subset=['year', 'raceName', 'driverName', 'grid', 'points', 'position'])

# Nettoyer les noms des pilotes pour assurer la cohérence
df['driverName'] = df['driverName'].str.strip().str.title()

# 3. Garder uniquement les vainqueurs pour une classification multi-classe
df_winners = df[df['position'] == 1].copy()

# Nettoyer les noms dans df_winners également
df_winners['driverName'] = df_winners['driverName'].str.strip().str.title()

# 4. Ajouter les performances de l'année précédente
# Créer une copie décalée des données pour obtenir les performances de l'année précédente
df_prev_year = df_winners.copy()
df_prev_year['year'] += 1  # Décaler l'année pour correspondre à la course actuelle avec l'année précédente
df_prev_year = df_prev_year[['year', 'raceName', 'driverName', 'position', 'points']]
df_prev_year.rename(columns={
    'position': 'prev_year_position',
    'points': 'prev_year_points'
}, inplace=True)

# Nettoyer les noms dans df_prev_year
df_prev_year['driverName'] = df_prev_year['driverName'].str.strip().str.title()

# Fusionner les données actuelles avec les performances de l'année précédente
df_winners = pd.merge(df_winners, df_prev_year, on=['year', 'raceName', 'driverName'], how='left')

# Remplacer les valeurs manquantes (si un pilote n'a pas participé l'année précédente)
df_winners['prev_year_position'] = df_winners['prev_year_position'].fillna(-1)  # -1 indique absence
df_winners['prev_year_points'] = df_winners['prev_year_points'].fillna(0)

# 5. Ajouter des caractéristiques agrégées pour les pilotes
# Calculer des statistiques agrégées pour chaque pilote
driver_stats = df.groupby('driverName').agg(
    avg_position=('position', 'mean'),
    total_points=('points', 'sum'),
    num_participations=('position', 'count')
).reset_index()

# Fusionner ces statistiques avec le DataFrame principal
df_winners = pd.merge(df_winners, driver_stats, on='driverName', how='left')

# Vérifier les colonnes après fusion
print("\nColonnes de df_winners après fusion et agrégation:")
print(df_winners.columns)

# Vérifier quelques lignes pour s'assurer que 'avg_position' est présente
print("\nExemple de données après fusion:")
print(df_winners[['driverName', 'avg_position', 'total_points', 'num_participations']].head())

# Vérifier si tous les pilotes sont présents dans driver_stats
missing_drivers = set(df_winners['driverName']) - set(driver_stats['driverName'])
if missing_drivers:
    print(f"\nPilotes manquants dans driver_stats: {missing_drivers}")
else:
    print("\nTous les pilotes sont présents dans driver_stats.")

# 6. Sélectionner les caractéristiques pertinentes (exclure 'driverName' car c'est la cible)
features = [
    'year',
    'raceName',
    'grid',
    'points',
    'prev_year_position',
    'prev_year_points',
    'avg_position',
    'total_points',
    'num_participations'
]
X = df_winners[features]
y = df_winners['driverName']  # Variable cible multi-classe

# 7. Assurer que les colonnes numériques sont de type numérique
numeric_features = ['year', 'grid', 'points', 'prev_year_position', 'prev_year_points', 'avg_position', 'total_points', 'num_participations']
X.loc[:, numeric_features] = X[numeric_features].apply(pd.to_numeric, errors='coerce')

# 8. Gérer les valeurs manquantes (s'il y en a encore)
# Imputer les valeurs manquantes pour les caractéristiques numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Imputer les valeurs manquantes pour les caractéristiques catégorielles et les encoder avec OneHotEncoder
categorical_features = ['raceName']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 9. Créer le préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 10. Créer le pipeline avec SMOTE et RandomForest
# Initialiser SMOTE avec k_neighbors=1
smote = SMOTE(random_state=42, k_neighbors=1)

# Initialiser le classificateur avec des poids équilibrés
classifier = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced'
)

# Créer le pipeline
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', classifier)
])

# 11. Séparer les données en ensembles d'entraînement et de test avec stratification
# Avant de faire cela, vérifier la distribution des classes
class_counts = y.value_counts()
print("\nDistribution des classes avant filtrage:")
print(class_counts)

# Filtrer les classes (pilotes) ayant au moins 2 exemples pour permettre la stratification
drivers_to_keep = class_counts[class_counts >= 2].index
df_winners_filtered = df_winners[df_winners['driverName'].isin(drivers_to_keep)].copy()

# Mettre à jour X et y après filtrage
X_filtered = df_winners_filtered[features]
y_filtered = df_winners_filtered['driverName']

# Assurer que les colonnes numériques sont de type numérique
X_filtered.loc[:, numeric_features] = X_filtered[numeric_features].apply(pd.to_numeric, errors='coerce')

# Vérifier la nouvelle distribution des classes
class_counts_filtered = y_filtered.value_counts()
print("\nDistribution des classes après filtrage (au moins 2 exemples par classe):")
print(class_counts_filtered)

# 12. Séparer les données en ensembles d'entraînement et de test avec stratification
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )
except ValueError as e:
    print(f"Erreur lors de la séparation des données : {e}")
    print("Assurez-vous que chaque classe a au moins deux exemples.")
    exit()

# 13. Vérifier la distribution des classes avant SMOTE
print("\nDistribution des classes dans y_train avant SMOTE:")
print(y_train.value_counts())

# 14. Entraîner le modèle avec SMOTE
pipeline.fit(X_train, y_train)

# 15. Faire des prédictions
predicted_classes = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)

# 16. Afficher les résultats des prédictions
print("\nExemples de prédictions :")
for i, (pred, prob) in enumerate(zip(predicted_classes, np.max(probabilities, axis=1)), 1):
    print(f"Exemple {i}: Pilote prédit = {pred}, Probabilité = {prob:.4f}")

# 17. Évaluer le modèle
print("\nRapport de Classification :")
print(classification_report(y_test, predicted_classes))

print("\nMatrice de Confusion :")
print(confusion_matrix(y_test, predicted_classes))

# 18. Évaluer l'importance des caractéristiques
model = pipeline.named_steps['classifier']

# Obtenir les noms des caractéristiques après OneHotEncoding
onehot_features = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(['raceName'])
all_features = numeric_features + list(onehot_features)
importances = model.feature_importances_

# Vérifier que le nombre d'importances correspond au nombre de caractéristiques
if len(importances) != len(all_features):
    print("\nAttention : Le nombre d'importances ne correspond pas au nombre de caractéristiques.")
    print(f"Nombre d'importances : {len(importances)}")
    print(f"Nombre de caractéristiques : {len(all_features)}")
else:
    feature_importances = pd.Series(importances, index=all_features).sort_values(ascending=False)
    print("\nImportance des Caractéristiques :")
    print(feature_importances)

# 19. Fonction pour prédire le vainqueur d'une course spécifique
def predict_top_n(race_year, race_name, top_n=3):
    """
    Prédit les top N pilotes pour une course spécifique en se basant sur les données disponibles.

    Parameters:
    - race_year (int) : Année de la course à prédire.
    - race_name (str) : Nom de la course à prédire.
    - top_n (int) : Nombre de pilotes à prédire.

    Returns:
    - DataFrame avec les top N pilotes et leurs probabilités d'être vainqueurs.
    """
    # Filtrer les pilotes ayant participé à la course
    race_data = df_winners_filtered[
        (df_winners_filtered['year'] == race_year) &
        (df_winners_filtered['raceName'] == race_name)
    ]

    if race_data.empty:
        print(f"Aucune donnée trouvée pour la course {race_name} en {race_year}.")
        return

    # Sélectionner les caractéristiques
    X_race = race_data[features]

    # Assurer que les colonnes numériques sont de type numérique
    X_race.loc[:, numeric_features] = X_race[numeric_features].apply(pd.to_numeric, errors='coerce')

    # Prédire les probabilités
    race_probabilities = pipeline.predict_proba(X_race)

    # Obtenir les noms des classes
    class_names = pipeline.named_steps['classifier'].classes_

    # Créer un DataFrame des probabilités
    prob_df = pd.DataFrame(race_probabilities, columns=class_names)
    prob_df['driverName'] = race_data['driverName'].values

    # Ajouter une colonne pour la probabilité maximale
    prob_df['prob_vainqueur'] = prob_df[class_names].max(axis=1)

    # Trier les pilotes par probabilité décroissante et sélectionner le top N
    prob_df_sorted = prob_df.sort_values(by='prob_vainqueur', ascending=False).head(top_n)

    return prob_df_sorted[['driverName', 'prob_vainqueur']]

# Exemple d'utilisation de la fonction
predicted_top3 = predict_top_n(2023, 'Bahrain Grand Prix', top_n=3)
print("\nTop 3 des pilotes prédits pour la course :")
print(predicted_top3)