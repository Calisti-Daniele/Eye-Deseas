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
df = pd.read_csv('../../datasets/dme/gila.csv', sep=";")

df['dr_level'] = df['dr_level'].str.strip()

# Applicare la funzione dalla 10ima colonna in poi
df = convert_strings_to_null(df, start_col=10)

df = encoding(df)
# Salvare il DataFrame aggiornato
df.to_csv("../../datasets/dme/prepared/gila.csv", sep=";", index=False)

print("Dataset aggiornato e salvato correttamente!")
