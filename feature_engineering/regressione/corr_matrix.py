import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions.utility_functions import *

# Caricare il dataset pulito
file_path = "../../datasets/dme/regressione/normalized/not_augmented/monthly.csv"
df = read_csv(file_path)

# Separare le feature per asse X e Y
y_features = df.columns  # Mostra tutte le feature sull'asse Y
x_features = df.columns[13:]  # Esclude le prime 13 feature per l'asse X
x_features = x_features[:-2]

# Calcolare la matrice di correlazione
correlation_matrix = df.corr()

# Selezionare solo le colonne richieste per l'asse X
correlation_matrix = correlation_matrix[x_features]

# Visualizzare la heatmap della correlazione
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, linewidths=0.5, yticklabels=y_features)
plt.title("Matrice di Correlazione tra le Feature")
plt.show()
