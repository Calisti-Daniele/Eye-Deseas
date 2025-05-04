import pandas as pd
import numpy as np
from functions.utility_functions import read_csv, save_csv

# === Feature statiche da includere in ogni step
STATIC_FEATURES = [
    'age', 'gender', 'insulinuser', 'smoker',
    'cataractsurgery', 'bi-lateral', 'height', 'weight', 'bmi'
]

def create_gru_dataset(path, start_n=3, target_n=20, save_path=None):
    """
    Prepara il dataset GRU con padding e masking.
    Ogni sequenza ha shape fissa = (target_n - 1, 1 + static), riempita con zeri.
    """
    print(f"📥 Caricamento del dataset da: {path}")
    df = read_csv(path).map(lambda x: pd.to_numeric(x, errors='coerce'))
    print(f"✅ Dataset caricato con shape: {df.shape}")

    max_len = target_n - 1  # tutte le sequenze saranno di questa lunghezza
    feature_dim = 1 + len(STATIC_FEATURES)

    X_list, y_list = [], []

    for idx, row in df.iterrows():
        static = row[STATIC_FEATURES].values.astype(np.float32)
        try:
            visits = [row[f'etdrs_{i}_visit'] for i in range(1, target_n + 1)]
        except KeyError:
            raise ValueError(f"❌ Manca una colonna etdrs nel file, riga {idx}")

        for n in range(start_n, target_n):
            # Sequenza reale
            x_seq = np.array(visits[:n], dtype=np.float32).reshape(-1, 1)  # (n, 1)
            static_rep = np.tile(static, (n, 1))                           # (n, static)
            x_full = np.concatenate([x_seq, static_rep], axis=1)          # (n, feature_dim)

            # Padding a max_len
            pad_len = max_len - n
            if pad_len > 0:
                x_full = np.pad(x_full, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)

            assert x_full.shape == (max_len, feature_dim)
            X_list.append(x_full)
            y_list.append(visits[n])  # etdrs_{n+1}_visit

    X = np.stack(X_list)  # (samples, max_len, feature_dim)
    y = np.array(y_list)

    print(f"📊 Dataset GRU costruito: X.shape={X.shape}, y.shape={y.shape}")

    if save_path:
        print(f"💾 Salvataggio su file: {save_path}")
        np.savez(save_path, X=X, y=y)
        print("✅ File salvato con successo")

    return X, y

# === Script entry point ===
if __name__ == "__main__":
    input_csv = "datasets/train_dataset_trex_not_augmented.csv"
    output_npz = "datasets/trex_dataset_trex_padded.npz"
    start_n = 3
    target_n = 20

    create_gru_dataset(input_csv, start_n=start_n, target_n=target_n, save_path=output_npz)
