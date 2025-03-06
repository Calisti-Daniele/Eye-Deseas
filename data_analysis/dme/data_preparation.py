import pandas as pd
import numpy as np

from functions.utility_functions import *


# start_col altrimenti anche il Gender me lo metterebbe a null
# Funzione utilizzata per eliminare i valori come "Missed"
def convert_strings_to_null(df: pd.DataFrame, start_col=6) -> pd.DataFrame:
    """
    Converte tutti i valori stringa in NaN, mantenendo i numeri invariati.
    Inoltre, converte numeri con virgola in float.
    """

    def clean_value(x):
        # Se è stringa e contiene una virgola, prova a convertirla in float
        if isinstance(x, str):
            x = x.replace(",", ".")  # Sostituisce la virgola con il punto
            try:
                return float(x)  # Converte in numero se possibile
            except ValueError:
                return np.nan  # Se non è un numero, lo trasforma in NaN
        return x  # Mantiene i numeri inalterati

    df.iloc[:, start_col:] = df.iloc[:, start_col:].map(clean_value)
    return df


def encoding(df: pd.DataFrame) -> pd.DataFrame:
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    vero_falso = ['insulinuser', 'smoker', 'cataractsurgery', 'bi-lateral']

    for col in vero_falso:
        df[col] = df[col].map({'VERO': 1, 'FALSO': 0})

    """
        Inactive PDR -> 0
        Mild -> 1
        Moderate -> 2
        Severe -> 3
    """
    df['dr_level'] = df['dr_level'].map({"Inactive PDR": 0, "Mild": 1, "Moderate": 2, "Severe": 3})

    df['dia-type'] = df['dia-type'].map({"Type 1": 1, "Type 2": 2})

    return df



# Caricamento del dataset
df = read_csv('../../datasets/dme/gila.csv')

df['dr_level'] = df['dr_level'].str.strip()

# Applicare la funzione dalla 10ima colonna in poi
df = convert_strings_to_null(df, start_col=10)

df = encoding(df)


#Elimino tutti i pazienti che non hanno dati finali

# Identificare le ultime due colonne del DataFrame
last_two_cols = df.columns[-2:]

# Filtrare il DataFrame eliminando le righe che hanno almeno un valore null nelle ultime due colonne
df_filtered = df.dropna(subset=last_two_cols, how='any')

df = df_filtered

#Elimino i pazienti che hanno meno di 1/3 delle visite
# Identificare le colonne dalla 14esima fino alla (i-2)esima colonna
columns_to_check = df.columns[13:-2]  # Indici basati su zero

# Calcolare la soglia massima di NaN consentiti (meno di 1/3 dei valori nulli)
threshold = len(columns_to_check) / 3

# Filtrare il DataFrame eliminando le righe che hanno più di 1/3 di valori null nelle colonne selezionate
df_filtered = df[df[columns_to_check].isnull().sum(axis=1) < threshold]

df = df_filtered

# Salvare il DataFrame aggiornato
save_csv(df, "../../datasets/dme/prepared/gila.csv")

print("Dataset aggiornato e salvato correttamente!")
