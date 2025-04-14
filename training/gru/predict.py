import numpy as np
import pandas as pd
import joblib
from keras.api.models import load_model
from functions.utility_functions import read_csv, save_csv

# === CONFIG ===
input_csv = "datasets/for_prediction/trex_10k_samples.csv"
output_csv = "datasets/prediction/trex_predicted_output_10k.csv"
model_path = "models/gru_etdrs_model_padded.keras"
scaler_X_path = "scaler/scaler_X_gru.pkl"
scaler_y_path = "scaler/scaler_y_gru.pkl"
start_n = 3        # etdrs_1 to etdrs_start_n
target_n = 15      # prediciamo etdrs_{start_n+1} fino a etdrs_target_n
mask_value = 0.0

# === STATIC FEATURES usate in training
STATIC_FEATURES = ['age', 'gender', 'insulinuser', 'smoker',
                   'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi']

# === FUNZIONE DI PREDIZIONE SU UNA RIGA
def predict_sequence_gru(row, model, scaler_X, scaler_y, start_n, target_n):
    static = row[STATIC_FEATURES].values.astype(np.float32)
    current_visits = []

    for i in range(1, start_n + 1):
        current_visits.append(row[f'etdrs_{i}_visit'])

    for step in range(start_n, target_n):
        seq_len = len(current_visits)
        x_seq = np.array(current_visits).reshape(-1, 1)
        static_rep = np.tile(static, (seq_len, 1))
        x_full = np.concatenate([x_seq, static_rep], axis=1)

        # Padding fino a target_n con mask_value
        if seq_len < target_n - 1:
            padding = np.full((target_n - 1 - seq_len, x_full.shape[1]), mask_value)
            x_full = np.concatenate([x_full, padding], axis=0)

        # Normalizzazione
        x_scaled = scaler_X.transform(x_full)
        x_input = x_scaled.reshape(1, target_n - 1, x_full.shape[1])

        # Predizione
        y_pred_norm = model.predict(x_input, verbose=0)[0][0]
        y_pred = scaler_y.inverse_transform([[y_pred_norm]])[0][0]

        current_visits.append(y_pred)
        row[f'etdrs_{step + 1}_visit'] = y_pred  # aggiorna anche il DataFrame

    return row

# === MAIN
if __name__ == "__main__":
    print(f"📥 Caricamento input da: {input_csv}")
    df = read_csv(input_csv).map(lambda x: pd.to_numeric(x, errors='coerce'))
    print(f"✅ Dataset caricato: {df.shape}")

    print(f"📦 Caricamento modello e scaler...")
    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    print(f"🧠 Inizio predizione da visita {start_n+1} a visita {target_n}...")
    df_pred = df.apply(lambda row: predict_sequence_gru(
        row, model, scaler_X, scaler_y, start_n, target_n
    ), axis=1)

    save_csv(df_pred, output_csv)
    print(f"✅ Predizioni salvate in: {output_csv}")
