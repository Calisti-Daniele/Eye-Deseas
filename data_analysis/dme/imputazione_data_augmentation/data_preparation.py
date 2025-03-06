import warnings
import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from functions.utility_functions import *

warnings.simplefilter(action='ignore', category=Warning)

# Carica il dataset
df = read_csv("../../../datasets/dme/prepared/gila.csv")

# Separare la prima colonna (esempio: 'study_id') e il resto dei dati
first_col = df.iloc[:, 0]  # Salva la prima colonna (ID del paziente)
df = df.iloc[:, 1:]  # Rimuove la prima colonna dal DataFrame

# Creazione del metadata per la data augmentation
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# Inizializzo il modello di sintesi dei dati
model = GaussianCopulaSynthesizer(metadata)

# Parametri per il ciclo
max_no_change_iters = 5  # Numero massimo di iterazioni senza miglioramenti
no_change_counter = 0  # Contatore iterazioni senza miglioramenti
prev_missing_count = df.isnull().sum().sum()  # Numero iniziale di valori nulli
iteration = 0
use_augmentation = True  # Alterna tra Data Augmentation e Riempimento con Media


def fill_na_with_adjacent_mean(row):
    """Sostituisce i valori NaN con la media dei valori adiacenti, se presenti."""
    row = row.copy()
    consecutive_nans = 0

    for i in range(1, len(row) - 1):
        if pd.isnull(row[i]):
            consecutive_nans += 1
            if consecutive_nans > 2:
                return row  # Se ci sono più di 2 NaN consecutivi, passa alla riga successiva

            left, right = None, None

            for j in range(i - 1, -1, -1):
                if not pd.isnull(row[j]):
                    left = row[j]
                    break

            for j in range(i + 1, len(row)):
                if not pd.isnull(row[j]):
                    right = row[j]
                    break

            if left is not None and right is not None:
                row[i] = (left + right) / 2
            elif left is not None:
                row[i] = left
            elif right is not None:
                row[i] = right
        else:
            consecutive_nans = 0

    return row


# **Itera alternando Data Augmentation e Riempimento con Media fino a eliminare i NaN**
while prev_missing_count > 0:
    iteration += 1
    print(f"Iterazione {iteration}: {prev_missing_count} valori nulli rimanenti")

    if use_augmentation:
        # **1️⃣ Data Augmentation con GaussianCopulaSynthesizer**
        print("Eseguo Data Augmentation...")

        threshold = int(len(df.columns) * 0.5)
        df_train = df.dropna(thresh=threshold)

        if df_train.empty:
            print("Dataset troppo incompleto, impossibile proseguire.")
            break

        model.fit(df_train)
        synth_data = model.sample(num_rows=len(df))
        df = df.combine_first(synth_data)

    else:
        # **2️⃣ Riempimento con Media tra Valori Adiacenti**
        print("Eseguo Riempimento con Media...")
        df = df.apply(fill_na_with_adjacent_mean, axis=1)

    # Controllo se il numero di NaN è cambiato
    current_missing_count = df.isnull().sum().sum()
    if current_missing_count == prev_missing_count:
        no_change_counter += 1
    else:
        no_change_counter = 0  # Reset contatore se ci sono miglioramenti

    if no_change_counter >= max_no_change_iters:
        print(
            f"Interrotto dopo {iteration} iterazioni: nessun miglioramento in {max_no_change_iters} cicli consecutivi.")
        break

    prev_missing_count = current_missing_count
    use_augmentation = not use_augmentation  # Alterna tra Data Augmentation e Riempimento con Media

# **Riaggiunge la prima colonna (ID del paziente)**
df.insert(0, first_col.name, first_col)

# **Salva il dataset aggiornato**
save_csv(df, "../../../datasets/dme/ready_to_use/imputazione_augmentation/gila.csv")

# Mostra i primi dati sintetici generati
print("Dati sintetici generati e dataset completato!")
print(df.head())
