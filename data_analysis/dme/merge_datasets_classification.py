import pandas as pd
from functions.utility_functions import *

# File e relativo nome del trattamento
file_mapping = {
    '../../datasets/dme/ready_to_use/augmented/gila_10k_samples.csv': 0, #gila
    '../../datasets/dme/ready_to_use/augmented/trex_10k_samples.csv': 1, #trex
    '../../datasets/dme/ready_to_use/augmented/monthly_10k_samples.csv': 2 #monthly
}

# Lista per salvare i dataframe
dataframes = []

# Ciclo sui file
for file, trattamento in file_mapping.items():
    # Carica il dataset
    df = read_csv(file)
    # Aggiunge la colonna "trattamento" con il nome corretto
    df['trattamento'] = trattamento
    # Determina il nome corretto della colonna finale da includere
    if 'etdrs_3_visit' in df.columns:
        stop_col = 'etdrs_3_visit'
    elif 'etdrs_visit_3' in df.columns:
        stop_col = 'etdrs_visit_3'
    else:
        raise ValueError(f"Colonna 'etdrs_3_visit' o 'etdrs_visit_3' non trovata in {file}.")

    # Prendi solo le colonne fino a quella individuata
    cols_to_keep = df.loc[:, :stop_col].columns.tolist()
    df = df[cols_to_keep + ['trattamento']]

    # Rinomina le colonne etdrs visit in modo uniforme
    rename_map = {
        'etdrs_visit_1': 'etdrs_1',
        'etdrs_visit_2': 'etdrs_2',
        'etdrs_visit_3': 'etdrs_3',
        'etdrs_1_visit': 'etdrs_1',
        'etdrs_2_visit': 'etdrs_2',
        'etdrs_3_visit': 'etdrs_3',
        'age_at_enrollment': 'age',
        'age_at_enrolment': 'age'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Aggiunge il dataframe alla lista
    dataframes.append(df)

# Unisce tutti i dataframe
merged_df = pd.concat(dataframes, ignore_index=True)


# Salva il risultato in un nuovo CSV
save_csv(merged_df,'../../datasets/dme/classificazione/dataset_unificato.csv')

print("✅ Dataset unificato salvato come 'dataset_unificato.csv'.")