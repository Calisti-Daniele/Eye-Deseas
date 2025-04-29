import pandas as pd
from functions.utility_functions import *

# Carica il dataset
df = read_csv('../../datasets/dme/classificazione/dataset_unificato.csv')

# Campiona 15 righe casuali mantenendo l'indice originale
sample_df = df.sample(n=30000, random_state=42)

# Salva gli indici per tracciabilità
sample_df['original_index'] = sample_df.index

# Rimuove colonne non numeriche e la colonna 'trattamento'
sample_clean = sample_df.select_dtypes(include=['number'])
if 'trattamento' in sample_clean.columns:
    sample_clean = sample_clean.drop(columns=['trattamento'])

# Salva il risultato
save_csv(sample_clean, 'for_prediction.csv')

# Salva anche una versione con i dati originali per confronto
save_csv(sample_df, 'for_prediction_with_info.csv')

print("✅ Campione da 15 righe salvato in 'for_prediction.csv'")
print("ℹ️  Versione con info complete salvata in 'for_prediction_with_info.csv'")
