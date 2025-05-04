import pandas as pd
from functions.utility_functions import *
# Carica i due file
reale = read_csv('for_prediction_with_info.csv')
predetto = read_csv('predizioni_con_trattamento_senza visite.csv')

# Assicurati che siano allineati
assert len(reale) == len(predetto), "❌ I file non hanno lo stesso numero di righe!"

# Confronta reale vs predetto
giuste = (reale['trattamento'] == predetto['trattamento_predetto']).sum()
totale = len(reale)
percentuale = (giuste / totale) * 100

print(f"✅ Predizioni corrette: {giuste}/{totale} ({percentuale:.2f}%)")

# Se vuoi anche il dettaglio riga per riga:
confronto = pd.DataFrame({
    'original_index': reale['original_index'],
    'trattamento_reale': reale['trattamento'],
    'trattamento_predetto': predetto['trattamento_predetto'],
    'corretto': reale['trattamento'] == predetto['trattamento_predetto']
})

save_csv(confronto, 'confronto_predizioni.csv')
print("📄 Dettaglio salvato in 'confronto_predizioni.csv'")
