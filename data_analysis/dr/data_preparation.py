import pandas as pd
import numpy as np


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


def gender_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})

    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_strings_to_null(df, start_col=4)

    df = gender_encoding(df)

    return df


# Caricamento del dataset
df = pd.read_csv('../../datasets/dr/OCT-DR_36W.csv', sep=";")

# Applicare la funzione dalla settima colonna in poi
df = prepare_df(df)

# Salvare il DataFrame aggiornato
df.to_csv("../../datasets/dr/prepared/OCT-DR_36W.csv", sep=";", index=False)

# Caricamento del dataset
df = pd.read_csv('../../datasets/dr/OCT-DR_104W.csv', sep=";")

df = prepare_df(df)

# Salvare il DataFrame aggiornato
df.to_csv("../../datasets/dr/prepared/OCT-DR_104W.csv", sep=";", index=False)

print("Dataset aggiornato e salvato correttamente!")
