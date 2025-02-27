import pandas as pd
from plot_functions.plot_func import *

# Caricamento del dataset
df = pd.read_csv('../../datasets/dr/prepared/OCT-DR_36W.csv', sep=";")

# Prendiamo gli ID unici dei pazienti
patients_id = df['patient_id'].unique()

# Dizionario per salvare le colonne con NaN per ogni paziente
patient_dict = {}

for patient in patients_id:
    row = df.loc[df['patient_id'] == patient]

    # Assicuriamoci che il paziente sia nel dizionario con un array vuoto
    if patient not in patient_dict:
        patient_dict[patient] = []

    for col in row.columns:
        if pd.isna(row[col].values[0]):  # Controlla se il valore è NaN
            patient_dict[patient].append(col)  # Aggiunge la colonna alla lista

for patient in patient_dict:
    print(f"{patient} - {len(patient_dict[patient])} null values")
