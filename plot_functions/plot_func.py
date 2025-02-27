import matplotlib.pyplot as plt
import pandas as pd

def bar_plot_gender(df: pd.DataFrame):
    # Conta la frequenza di M e F
    gender_counts = df["gender"].value_counts()

    # Creazione del grafico a barre
    plt.figure(figsize=(6, 4))
    plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'], alpha=0.7)

    # Aggiungi etichette
    plt.xlabel("Genere")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione di Gender nel Dataset")


    # Mostra il grafico
    plt.show()

def bar_plot_age(df: pd.DataFrame):
    # Creazione del grafico a barre
    plt.figure(figsize=(6, 4))
    plt.bar(df['age'].index, df['age'].values, alpha=0.7)

    # Aggiungi etichette
    plt.xlabel("Età")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione di Età nel Dataset")

    # Mostra il grafico
    plt.show()

def plot_mean_values(df: pd.DataFrame):
    # Filtrare le colonne relative a CST e BCVA
    cst_columns = [col for col in df.columns if "cst_" in col]
    bcva_columns = [col for col in df.columns if "bcva_" in col]

    # Calcolare la media per ogni settimana
    cst_means = df[cst_columns].apply(pd.to_numeric, errors='coerce').mean()

    bcva_means = df[bcva_columns].apply(pd.to_numeric, errors='coerce').mean()

    # Estrarre le settimane
    weeks = [int(col.split("_")[1][:-1]) for col in cst_columns]

    # Creare il grafico per CST
    plt.figure(figsize=(10, 5))
    plt.plot(weeks, cst_means, marker='o', linestyle='-', color='blue', label="CST Media")
    plt.xlabel("Settimane")
    plt.ylabel("CST Medio")
    plt.title("Variazione del CST nel tempo")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Creare il grafico per BCVA
    plt.figure(figsize=(10, 5))
    plt.plot(weeks, bcva_means, marker='s', linestyle='-', color='red', label="BCVA Medio")
    plt.xlabel("Settimane")
    plt.ylabel("BCVA Medio")
    plt.title("Variazione del BCVA nel tempo")
    plt.legend()
    plt.grid(True)
    plt.show()