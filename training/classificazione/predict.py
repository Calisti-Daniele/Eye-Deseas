import pandas as pd
import pickle

from functions.utility_functions import read_csv, save_csv

# Carica il modello
with open('catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Carica nuovi dati
nuovi_dati = read_csv('for_prediction.csv')  # <-- cambia con il tuo file
X_nuovi = nuovi_dati.select_dtypes(include=['number'])  # solo colonne numeriche

print("Colonne presenti nel file:", nuovi_dati.columns.tolist())
print("Colonne numeriche trovate:", X_nuovi.columns.tolist())

# Predizione
predizioni = model.predict(X_nuovi)

# Output
nuovi_dati['trattamento_predetto'] = predizioni
save_csv(nuovi_dati,'predizioni_con_trattamento.csv')

print("✅ Predizioni salvate in 'predizioni_con_trattamento.csv'")
