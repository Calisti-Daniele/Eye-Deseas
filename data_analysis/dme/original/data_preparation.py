import pandas as pd
import numpy as np
from functions.utility_functions import *

# Carica il dataset
df = read_csv("../../../datasets/dme/prepared/monthly.csv")

# Funzione per riempire i valori NaN con la media tra i valori adiacenti
def fill_na_with_adjacent_mean(row):
    row = row.copy()  # Evita modifiche dirette alla riga originale
    for i in range(1, len(row) - 1):  # Evita il primo e l'ultimo valore
        if pd.isnull(row[i]):  # Se il valore è NaN
            left, right = None, None
            # Trova il primo valore a sinistra
            for j in range(i - 1, -1, -1):
                if not pd.isnull(row[j]):
                    left = row[j]
                    break
            # Trova il primo valore a destra
            for j in range(i + 1, len(row)):
                if not pd.isnull(row[j]):
                    right = row[j]
                    break
            # Se entrambi i valori esistono, calcola la media
            if left is not None and right is not None:
                row[i] = (left + right) / 2
            elif left is not None:
                row[i] = left  # Se c'è solo il valore a sinistra
            elif right is not None:
                row[i] = right  # Se c'è solo il valore a destra
    return row

# Applicare la funzione a tutto il DataFrame
df_filled = df.apply(fill_na_with_adjacent_mean, axis=1)

# Salvare il dataset aggiornato
save_csv(df_filled,"../../../datasets/dme/ready_to_use/imputazione/monthly.csv")

# Mostrare le prime righe per confermare
print(df_filled.head())
