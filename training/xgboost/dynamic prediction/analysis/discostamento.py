import pandas as pd
from functions.utility_functions import *

# Carica i due file CSV
df1 = read_csv("../../../../datasets/dme/ready_to_use/augmented/trex_10k_samples.csv")[:99]
df2 = read_csv("../datasets/prediction/trex_predicted_output_10k.csv")[:99]

# Assicurati che abbiano lo stesso numero di righe
if len(df1) != len(df2):
    raise ValueError("I due file hanno un numero diverso di righe")

# Calcola i valori e il discostamento
risultato = pd.DataFrame({
    'valore_reale': df1['etdrs_20_visit'],
    'valore_predetto': df2['etdrs_20_visit'],
})

# Calcolo del discostamento assoluto
risultato['discostamento'] = abs(risultato['valore_reale'] - risultato['valore_predetto'])

# Accuracy personalizzata (discostamento < 5)
accuracy_personalizzata = (risultato['discostamento'] < 5).mean() * 100

# Analisi finale del discostamento
analisi = {
    'media_discordanza': risultato['discostamento'].mean(),
    'mediana_discordanza': risultato['discostamento'].median(),
    'deviazione_standard': risultato['discostamento'].std(),
    'massimo_discordanza': risultato['discostamento'].max(),
    'minimo_discordanza': risultato['discostamento'].min(),
    'percentuale_sotto_5': (risultato['discostamento'] < 5).mean() * 100,
    'percentuale_sotto_10': (risultato['discostamento'] < 10).mean() * 100,
    'accuracy_personalizzata_(<5)': accuracy_personalizzata
}

def etichetta_dme(x):
    if x <= 33:
        return "diffuse"
    elif x < 67:
        return "intermediate"
    else:
        return "focal"

# Aggiungi le etichette di classe al DataFrame
risultato['classe_reale'] = risultato['valore_reale'].apply(etichetta_dme)
risultato['classe_predetta'] = risultato['valore_predetto'].apply(etichetta_dme)

# Accuracy di classificazione
accuracy_classificazione = (risultato['classe_reale'] == risultato['classe_predetta']).mean() * 100
print(f"\nAccuracy classificazione DME: {accuracy_classificazione:.2f}%")


print("\n--- Analisi del Discostamento ---")
for k, v in analisi.items():
    print(f"{k.replace('_', ' ').capitalize()}: {v:.2f}%"
          if "percentuale" in k or "accuracy" in k else f"{k.replace('_', ' ').capitalize()}: {v:.2f}")

# (Opzionale) Salva su file
save_csv(risultato, "confronto_etdrs_20_10k.csv")
