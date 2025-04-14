import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from functions.utility_functions import *

# Dizionario con i percorsi dei dataset
datasets = {
    'trex': '../datasets/dme/ready_to_use/imputazione/trex.csv',
    'gila': '../datasets/dme/ready_to_use/imputazione/gila.csv',
    'monthly': '../datasets/dme/ready_to_use/imputazione/monthly.csv'
}

for key, value in datasets.items():
    # Carica il dataset reale
    data = read_csv(value)

    # Visualizza le prime righe del dataset
    print(f"Dataset {key} - Prime righe:")
    print(data.head())

    # Crea la metadata del dataset
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    # Inizializza il sintetizzatore con la metadata
    synthesizer = GaussianCopulaSynthesizer(metadata)

    # Addestra il sintetizzatore
    synthesizer.fit(data)

    # Genera dati sintetici
    synthetic_data = synthesizer.sample(num_rows=10000)

    # Converte tutti i valori numerici in interi (senza cifre decimali)
    for col in synthetic_data.select_dtypes(include=['float', 'int']).columns:
        synthetic_data[col] = synthetic_data[col].round().astype(int)

    # Salva il dataset sintetico
    save_csv(synthetic_data, f"../datasets/dme/ready_to_use/augmented/{key}_10k_samples.csv")

    print(f"Dataset sintetico {key} salvato con valori numerici convertiti in interi.")
