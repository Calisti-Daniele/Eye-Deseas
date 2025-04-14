import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.stats import linregress
from functions.utility_functions import read_csv, save_csv

# === Fourier e trend ===
def fourier_transform(row):
    return np.abs(fft(row.values))[1]  # primo coeff. utile

def compute_trend(row):
    x = np.arange(len(row))
    y = row.values
    slope, _, _, _, _ = linregress(x, y)
    return slope

# === Colonne statiche
STATIC_FEATURES = [
    'age', 'gender', 'insulinuser', 'smoker',
    'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi'
]

# === Funzione di generazione
def generate_dynamic_dataset(data, min_n=3, max_n=20):
    dynamic_data = []

    for _, row in data.iterrows():
        for n in range(min_n, max_n + 1):
            row_dict = {}

            # Visite etdrs_1 → etdrs_n
            for i in range(1, n + 1):
                row_dict[f'etdrs_{i}_visit'] = row[f'etdrs_{i}_visit']

            # Statiche
            for feat in STATIC_FEATURES:
                row_dict[feat] = row[feat]

            # Fourier: su 20 valori (padded con 0)
            padded = [row.get(f'etdrs_{i}_visit', 0) for i in range(1, 21)]
            row_dict['fourier_coeff'] = fourier_transform(pd.Series(padded))

            # Trend: solo sulle visite effettive
            actual = [row[f'etdrs_{i}_visit'] for i in range(1, n + 1)]
            row_dict['trend'] = compute_trend(pd.Series(actual))

            # n (posizione temporale)
            row_dict['n'] = n

            # Target → visita n+1
            row_dict['target'] = row[f'etdrs_{n + 1}_visit']

            dynamic_data.append(row_dict)

    return pd.DataFrame(dynamic_data)

# === MAIN
if __name__ == "__main__":
    input_path = "../../../datasets/dme/ready_to_use/augmented/trex_10k_samples.csv"  # CSV completo con tutte le visite
    output_path = "datasets/dynamic_dataset_trex_10k.csv"

    data = read_csv(input_path).map(lambda x: pd.to_numeric(x, errors='coerce'))
    print(f"✅ Dataset originale caricato: {data.shape}")

    dynamic_df = generate_dynamic_dataset(data, min_n=3, max_n=19)
    print(f"📈 Dataset dinamico generato: {dynamic_df.shape}")

    save_csv(dynamic_df, output_path)
    print(f"✅ Salvato in: {output_path}")
