import pandas as pd
import numpy as np
from functions.utility_functions import read_csv, save_csv

STATIC_FEATURES = ['age', 'gender', 'insulinuser', 'smoker',
                   'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi']

def prepare_random_gru_dataset(input_path, start_n=3, target_n=20, save_path=None):
    df = read_csv(input_path).map(lambda x: pd.to_numeric(x, errors='coerce'))

    X_list, y_list = [], []

    for _, row in df.iterrows():
        static = row[STATIC_FEATURES].values.astype(np.float32)
        visits = [row.get(f'etdrs_{i}_visit', np.nan) for i in range(1, target_n + 1)]

        if any(pd.isna(visits)): continue  # scarta righe incomplete

        for n in range(start_n, target_n):
            x_seq = np.array(visits[:n], dtype=np.float32).reshape(-1, 1)
            static_rep = np.tile(static, (n, 1))
            x_full = np.concatenate([x_seq, static_rep], axis=1)  # (n, features)

            X_list.append(x_full)
            y_list.append(visits[n])  # etdrs_{n+1}_visit

    # Pad sequences to max length
    max_len = target_n - 1
    feature_size = X_list[0].shape[1]
    X_padded = np.zeros((len(X_list), max_len, feature_size), dtype=np.float32)

    for i, seq in enumerate(X_list):
        X_padded[i, :seq.shape[0], :] = seq

    y_array = np.array(y_list, dtype=np.float32)

    if save_path:
        np.savez(save_path, X=X_padded, y=y_array)

    print(f"✅ Dataset salvato con shape: {X_padded.shape}, y: {y_array.shape}")
    return X_padded, y_array

if __name__ == "__main__":
    prepare_random_gru_dataset(
        "../../../datasets/dme/ready_to_use/augmented/trex_10k_samples.csv",
        save_path="datasets/gru_dataset_random_10k.npz"
    )
