import pandas as pd
import numpy as np
import joblib
from scipy.fft import fft
from scipy.stats import linregress
import os
from functions.utility_functions import read_csv, save_csv

# === Fourier e trend ===
def fourier_transform(row):
    return np.abs(fft(row.values))[1]

def compute_trend(row):
    x = np.arange(len(row))
    y = row.values
    slope, _, _, _, _ = linregress(x, y)
    return slope

STATIC_FEATURES = [
    'age', 'gender', 'insulinuser', 'smoker',
    'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi'
]

# === Predizione iterativa su una riga
def predict_row_iterative_models(row, start_n, target_n, models_dir, features_dir):
    current_visits = [row[f'etdrs_{i}_visit'] for i in range(1, start_n + 1)]
    static = {feat: row[feat] for feat in STATIC_FEATURES}

    for n in range(start_n, target_n):
        model_path = os.path.join(models_dir, f"xgb_n{n}_10k.pkl")
        feat_path = os.path.join(features_dir, f"features_n{n}_10k.pkl")

        if not os.path.exists(model_path) or not os.path.exists(feat_path):
            print(f"⚠️ Modello o feature non trovati per n={n}, salto")
            break

        model = joblib.load(model_path)
        feature_names = joblib.load(feat_path)

        # === Prepara input
        input_dict = {f'etdrs_{i}_visit': current_visits[i - 1] for i in range(1, n + 1)}
        input_dict.update(static)

        padded = np.array(current_visits + [0.0] * (20 - len(current_visits)))
        input_dict['fourier_coeff'] = fourier_transform(pd.Series(padded))
        input_dict['trend'] = compute_trend(pd.Series(current_visits))
        input_dict['n'] = n

        # Ricostruisci X con tutte le feature attese, riempiendo i buchi con 0
        full_input = {f: input_dict.get(f, 0.0) for f in feature_names}
        X = pd.DataFrame([full_input], columns=feature_names)

        # Predizione
        pred = model.predict(X)[0]
        visit_col = f'etdrs_{n + 1}_visit'
        current_visits.append(pred)
        row[visit_col] = pred  # Aggiunge anche al DataFrame finale

    return row

# === MAIN
if __name__ == "__main__":
    input_csv = "datasets/for_prediction/trex_10k_samples.csv"   # Con etdrs_1..etdrs_start
    output_csv = "datasets/prediction/trex_predicted_output_10k.csv"
    models_dir = "models/by_n/"
    features_dir = "scaler/by_n/"
    start_n = 3
    target_n = 20

    # === Carica dati
    df = read_csv(input_csv).map(lambda x: pd.to_numeric(x, errors='coerce'))
    print(f"✅ Input caricato: {df.shape}")

    df_pred = df.apply(lambda row: predict_row_iterative_models(
        row, start_n, target_n, models_dir, features_dir
    ), axis=1)

    save_csv(df_pred, output_csv)
    print(f"✅ Predizioni salvate fino a etdrs_{target_n}_visit in: {output_csv}")
