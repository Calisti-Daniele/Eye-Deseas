import pandas as pd
from functions.plot_func import *

datasets = {
    'trex': '../../datasets/dme/ready_to_use/imputazione/trex.csv',
    'gila': '../../datasets/dme/ready_to_use/imputazione/gila.csv',
    'monthly': '../../datasets/dme/ready_to_use/imputazione/monthly.csv'
}

results = {
    'trex': [],
    'gila': [],
    'monthly': [],
}

null_values = {
    'trex': 0,
    'gila': 0,
    'monthly': 0,
}

for key,value in datasets.items():
    # Caricamento del dataset
    df = pd.read_csv(value, sep=";")

    print(f"-------------------------{key}--------------------------")

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
        missed_visits = len(patient_dict[patient])/3
        null_visits = len(patient_dict[patient])
        results[key].append(missed_visits)
        null_values[key] += len(patient_dict[patient])
        print(f"{patient} - {null_visits} null values --> {missed_visits} visite mancanti")


for key,value in results.items():
    print(f"\n{key}")
    print(f" Valori nulli: {null_values[key]}\n")