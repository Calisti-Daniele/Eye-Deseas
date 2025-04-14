import numpy as np
import pandas as pd
import joblib
from keras.api.models import load_model
from functions.utility_functions import read_csv, save_csv

STATIC_FEATURES = ['age', 'gender', 'insulinuser', 'smoker',
                   'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi']

def prepare_input_sequence(current_visits, static_values, total_timesteps=19):
    """
    Costruisce una sequenza GRU a partire dalle visite e dalle feature statiche.
    Fa il padding fino a `total_timesteps` con zeri.
    """
    n = len(current_visits)
    x_seq = np.array(current_visits, dtype=np.float32).reshape(-1, 1)              # (n, 1)
    static_rep = np.tile(static_values, (n, 1))                                     # (n, static)
    x_combined = np.concatenate([x_seq, static_rep], axis=1)                        # (n, 1 + static)

    # Pad con zeri
    pad_len = total_timesteps - n
    if pad_len > 0:
        padding = np.zeros((pad_len, x_combined.shape[1]), dtype=np.float32)
        x_combined = np.vstack([x_combined, padding])

    return x_combined  # (19, features)


def predict_sequence_gru(row, model, scaler_y, start_n=3, target_n=20):
    """
    Predice le visite da n+1 a target_n usando la GRU e gestisce la propagazione dell’errore.
    """
    current_visits = [row[f'etdrs_{i}_visit'] for i in range(1, start_n + 1)]
    static = row[STATIC_FEATURES].values.astype(np.float32)

    for t in range(start_n + 1, target_n + 1):
        X_seq = prepare_input_sequence(current_visits, static)  # (19, features)
        X_input = np.expand_dims(X_seq, axis=0)                 # (1, 19, features)

        y_pred_norm = model.predict(X_input, verbose=0)[0][0]
        y_pred = scaler_y.inverse_transform([[y_pred_norm]])[0][0]  # Denormalizzazione

        current_visits.append(y_pred)
        row[f'etdrs_{t}_visit'] = y_pred

    return row


# === MAIN ===
if __name__ == "__main__":
    input_csv = "datasets/for_prediction/trex_10k_samples.csv"
    output_csv = "datasets/prediction/trex_gru_predicted_10k_bidirectional.csv"
    model_path = "models/gru_scheduled_10k_bidirectional.keras"
    scaler_path = "scaler/scaler_gru_y_10k_bidirectional.pkl"
    start_n = 3
    target_n = 20

    print("📥 Caricamento dati e modello...")
    df = read_csv(input_csv).map(lambda x: pd.to_numeric(x, errors='coerce'))
    model = load_model(model_path)
    scaler_y = joblib.load(scaler_path)

    print(f"🔮 Predizione da visita {start_n+1} a {target_n} per {len(df)} pazienti...")

    df_pred = df.apply(lambda row: predict_sequence_gru(
        row, model, scaler_y, start_n=start_n, target_n=target_n
    ), axis=1)

    save_csv(df_pred, output_csv)
    print(f"✅ Output salvato in: {output_csv}")
