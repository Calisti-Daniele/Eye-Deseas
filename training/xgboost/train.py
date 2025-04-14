import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from functions.utility_functions import *

# Caricamento del dataset
# Caricamento del dataset
# Caricamento del dataset
data = read_csv('feature_selection/train_dataset_trex_not_augmented_normalized.csv')
data = data.map(convert_to_float)  # Converti tutti i valori nel formato corretto

# Trova l'indice della colonna target
target_column = "etdrs_19_visit"
target_index = data.columns.get_loc(target_column)

# Mantieni tutte le feature prima di "etdrs_3_visit" + le prime 3 visite e il target "etdrs_20_visit"
selected_features = list(data.columns[:target_index + 1]) + ["etdrs_20_visit", "fourier_coeff", "trend"]
data = data[selected_features]

# Separazione tra feature (X) e target (y)
X = data.drop("etdrs_20_visit", axis=1)  # Manteniamo tutte le feature selezionate tranne il target
y = data["etdrs_20_visit"]  # Target: predire la 20ª visita

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione del modello con i migliori iperparametri
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=200,
    random_state=42,
    booster='gbtree',  # Booster predefinito
)

# Ottimizzatori alternativi
booster_options = ['gbtree', 'gblinear', 'dart']

# Training con diversi ottimizzatori
best_rmse = float("inf")
best_booster = None
best_model = None

for booster in booster_options:
    print(f"\nTraining con booster: {booster}")
    model.set_params(booster=booster)
    model.fit(X_train, y_train)

    # Valutazione sul test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE con booster {booster}: {rmse:.4f}")

    # Salvare il miglior modello
    if rmse < best_rmse:
        best_rmse = rmse
        best_booster = booster
        best_model = model

# Salvataggio del miglior modello
model_save_path = "../../models/best_xgboost_model_not_augmented.pkl"
joblib.dump(best_model, model_save_path)
print(f"\nMiglior modello salvato in: {model_save_path}")
print(f"Miglior booster: {best_booster}, RMSE: {best_rmse:.4f}")

# Valutazione finale del miglior modello
y_final_pred = best_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
print(f"\nValutazione finale sul test set con il miglior modello:")
print(f"RMSE: {final_rmse:.4f}")
