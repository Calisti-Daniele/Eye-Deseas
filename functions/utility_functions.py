import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost.testing.data import joblib


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, sep=";", index=False)


def normalizza_feature_numeriche(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    # Normalizzazione delle feature numeriche
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    joblib.dump(scaler, scaler_path)  # Salva lo scale

    return df


def convert_to_float(value):
    """Rimuove separatori di migliaia e converte in float."""
    if isinstance(value, str):
        value = value.replace('.', '')  # Rimuove separatori di migliaia
        value = value.replace(',', '.')  # Sostituisce la virgola con il punto per i decimali
        try:
            return float(value)
        except ValueError:
            return np.nan  # Se non convertibile, restituisce NaN
    return value  # Se già numerico, lo lascia invariato