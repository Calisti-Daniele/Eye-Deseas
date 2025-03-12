import pandas as pd
from functions.utility_functions import read_csv

# Dizionario con i percorsi dei dataset
datasets = {
    'trex': '../../datasets/dme/ready_to_use/augmented/trex.csv',
    'gila': '../../datasets/dme/ready_to_use/augmented/gila.csv',
    'monthly': '../../datasets/dme/ready_to_use/augmented/monthly.csv'
}

# Dizionario per salvare i risultati del describe
describe_results = {}

# Caricare e calcolare il describe per ogni dataset
for key, value in datasets.items():
    try:
        df = read_csv(value)  # Legge il dataset
        df_subset = df.iloc[:, 34:]  # Seleziona dalla 10ª colonna in poi (0-based index)
        describe_results[key] = df_subset.describe(include='all')  # Calcola il describe
    except Exception as e:
        describe_results[key] = f"Errore nel caricamento del dataset {key}: {e}"

# Stampare i risultati in modo formattato
for key, df_desc in describe_results.items():
    print(f"\n---------- Describe {key} ----------\n")
    if isinstance(df_desc, pd.DataFrame):
        print(df_desc.to_string())  # Migliore leggibilità con to_string()
    else:
        print(df_desc)
