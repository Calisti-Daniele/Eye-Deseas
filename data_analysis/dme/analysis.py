import pandas as pd
from plot_functions.plot_func import *

# Caricamento del dataset
df = pd.read_csv('../../datasets/dme/prepared/trex.csv', sep=";")

# Prendiamo gli ID unici dei pazienti
patients_id = df['study_id'].unique()

# Dizionario per salvare le colonne con NaN per ogni paziente
patient_dict = {}

for patient in patients_id:
    row = df.loc[df['study_id'] == patient]

    # Assicuriamoci che il paziente sia nel dizionario con un array vuoto
    if patient not in patient_dict:
        patient_dict[patient] = []

    for col in row.columns:
        #print(row[col].name)
        if pd.isna(row[col].values[0]):  # Controlla se il valore è NaN
            patient_dict[patient].append(col)  # Aggiunge la colonna alla lista

for patient in patient_dict:
    print(f"{patient} - {len(patient_dict[patient])} null values")
