import pandas as pd
from functions.utility_functions import *
def convert_numeric_to_int_float(df):
    """
    Converte tutte le colonne numeriche in float senza decimali.
    """
    df = df.copy()  # Evita modifiche all'originale
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = df[col].astype(float).round(0)  # Arrotonda e mantiene float senza decimali
    return df

# Esempio di utilizzo
df = read_csv('datasets/dme/regressione/normalized/augmented/monthly.csv')

df_transformed = convert_numeric_to_int_float(df)

save_csv(df_transformed, 'datasets/dme/regressione/normalized/augmented/monthly.csv')
