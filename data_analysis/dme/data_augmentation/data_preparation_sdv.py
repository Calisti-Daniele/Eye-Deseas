import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from functions.utility_functions import *

# Carica il dataset
df = read_csv("../../../datasets/dme/prepared/trex.csv")

# Separare la prima colonna (esempio: 'study_id') e il resto dei dati
first_col = df.iloc[:, 0]  # Salva la prima colonna (ID del paziente)
df = df.iloc[:, 1:]  # Rimuove la prima colonna dal DataFrame

# Creazione del metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# Inizializzo il modello di sintesi dei dati
model = GaussianCopulaSynthesizer(metadata)

# **Itera finché ci sono valori nulli nel dataset o finché il numero di NaN non cambia per N iterazioni**
max_no_change_iters = 5  # Numero massimo di iterazioni senza miglioramenti
no_change_counter = 0  # Contatore iterazioni senza miglioramenti
prev_missing_count = df.isnull().sum().sum()  # Numero iniziale di valori nulli

iteration = 0
while prev_missing_count > 0:
    iteration += 1
    print(f"Iterazione {iteration}: {prev_missing_count} valori nulli rimanenti")

    # **1️⃣ Selezioniamo solo le righe con almeno il 50% di dati completi**
    threshold = int(len(df.columns) * 0.5)  # Almeno il 50% dei dati non deve essere NaN
    df_train = df.dropna(thresh=threshold)  # Mantiene solo le righe con abbastanza dati

    # Se anche così è vuoto, significa che il dataset è troppo incompleto
    if df_train.empty:
        print("Dataset troppo incompleto, impossibile proseguire.")
        break

    # **2️⃣ Alleno il modello su dati reali con meno NaN**
    model.fit(df_train)

    # **3️⃣ Generazione di Dati Sintetici per l'intero dataset**
    synth_data = model.sample(num_rows=10000)

    # **4️⃣ Riempie OGNI valore nullo con il valore corrispondente del dataset sintetico**
    df = df.combine_first(synth_data)

    # **5️⃣ Controllo se il numero di NaN è cambiato**
    current_missing_count = df.isnull().sum().sum()
    if current_missing_count == prev_missing_count:
        no_change_counter += 1
    else:
        no_change_counter = 0  # Reset contatore se ci sono miglioramenti

    # Se il numero di NaN non cambia per N iterazioni, interrompe il ciclo
    if no_change_counter >= max_no_change_iters:
        print(f"Interrotto dopo {iteration} iterazioni: nessun miglioramento in {max_no_change_iters} cicli consecutivi.")
        break

    prev_missing_count = current_missing_count  # Aggiorna il conteggio dei NaN

# **6️⃣ Riaggiunge la prima colonna (ID del paziente)**
df.insert(0, first_col.name, first_col)

# **7️⃣ Salva il dataset aggiornato**
save_csv(df, "../../../datasets/dme/ready_to_use/data_augmentation/monthly.csv")

# Mostra i primi dati sintetici generati
print("Dati sintetici generati e dataset completato!")
print(df.head())
