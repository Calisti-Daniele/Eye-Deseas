from functions.utility_functions import *
import joblib
import pandas as pd
import numpy as np

# Caricamento del modello salvato
model_path = "../../models/best_xgboost_model_not_augmented.pkl"
best_model = joblib.load(model_path)
print(f"Modello caricato da: {model_path}")

# Caricamento dello scaler
scaler_path = "../../feature_engineering/regressione/scaler/scaler_trex_not_augmented.pkl"  # Percorso dello scaler usato per la normalizzazione
scaler = joblib.load(scaler_path)
print(f"Scaler caricato da: {scaler_path} con num di feature: {scaler.n_features_in_}")

# Caricamento dei dati non visti
new_data_path = "feature_selection/train_dataset_trex_not_augmented_for_prediction.csv"  # Percorso corretto
new_data = read_csv(new_data_path)

# Creare una lista con i nomi delle feature originali (quelle usate per lo scaling)
feature_names = joblib.load("../../feature_engineering/regressione/scaler/feature_names_trex_not_augmented.pkl")  # Percorso con i nomi delle feature originali

print(feature_names)

new_data = new_data.map(convert_to_float)

# Effettuare le previsioni
predictions = best_model.predict(new_data)

# Creare un array di dummy con la stessa forma dei dati originali
dummy_input = np.zeros((predictions.shape[0], scaler.n_features_in_))  # n_features_in_ è il numero di colonne originali

# Creare un DataFrame temporaneo con colonne originali (in modo che l'ordine sia corretto)
dummy_df = pd.DataFrame(dummy_input, columns=feature_names)

# Inserire le predizioni nella colonna target
target_column = "etdrs_20_visit"
dummy_df[target_column] = predictions  # Inseriamo i valori predetti

# Applicare l'inverso della normalizzazione
dummy_output = scaler.inverse_transform(dummy_df)

# Estrarre solo la colonna predetta denormalizzata
predictions_denormalized = dummy_output[:, dummy_df.columns.get_loc(target_column)]

# Aggiungere le previsioni al DataFrame
new_data[target_column] = predictions_denormalized

# Salvataggio del risultato
output_path = "feature_selection/previsioni_dati_trex_non_visti.csv"
save_csv(new_data, output_path)
print(f"Previsioni salvate in: {output_path}")