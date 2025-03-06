import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, sep=";", index=False)
