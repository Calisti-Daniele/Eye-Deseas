from functions.utility_functions import *
import joblib
import pandas as pd
import numpy as np
from keras.api.models import load_model

# Percorsi dei file
model_path = "../../models/best_model_not_augmented_neural.h5"
scaler_path = "scaler/scaler_trex_not_augmented_neural.pkl"
feature_names_path = "scaler/feature_names_trex_not_augmented_neural.pkl"
new_data_path = "../xgboost/feature_selection/train_dataset_trex_not_augmented_for_prediction.csv"
output_path = "previsioni_dati_trex_non_visti.csv"

# Caricamento del modello
best_model = load_model(model_path)
print(f"Modello caricato da: {model_path}")

# Caricamento dello scaler e dei nomi delle feature
scaler = joblib.load(scaler_path)
feature_names = joblib.load(feature_names_path)

print(f"Scaler caricato da: {scaler_path} con num di feature: {scaler.n_features_in_}")
print(f"Feature usate: {feature_names}")

# Caricamento dei nuovi dati
new_data = read_csv(new_data_path)
new_data = new_data.map(convert_to_float)

# Estrazione delle feature nell'ordine corretto
X_to_predict = new_data[feature_names]
# Aggiungiamo una colonna placeholder per etdrs_20_visit
X_full = X_to_predict.copy()
X_full["etdrs_20_visit"] = 0  # valore dummy, verrà sovrascritto dopo

# Ora possiamo fare lo scaling
X_scaled = scaler.transform(X_full)

# Predizione con la rete neurale
predictions = best_model.predict(X_scaled).flatten()

# Denormalizzazione: costruiamo un dummy_df come nel tuo script
dummy_input = np.zeros((predictions.shape[0], scaler.n_features_in_))
dummy_df = pd.DataFrame(dummy_input, columns=feature_names)
dummy_df["etdrs_20_visit"] = predictions

# Inverso dello scaling
dummy_output = scaler.inverse_transform(dummy_df)

# Estrazione della colonna predetta denormalizzata
target_column = "etdrs_20_visit"
predictions_denormalized = dummy_output[:, dummy_df.columns.get_loc(target_column)]

# Inserimento nel dataframe originale
new_data[target_column] = predictions_denormalized

# Salvataggio del risultato
save_csv(new_data, output_path)
print(f"Previsioni salvate in: {output_path}")
