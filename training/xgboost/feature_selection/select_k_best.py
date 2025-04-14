from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
from functions.utility_functions import *

# Carica il dataset
data = read_csv('../../../datasets/dme/ready_to_use/augmented/trex.csv')

fourier_coeff = data['fourier_coeff']
trend = data['trend']

# Trova l'indice della colonna target
target_column = "etdrs_20_visit"
target_index = data.columns.get_loc(target_column)

# Mantieni solo le colonne fino a "etdrs_20_visit"
data = data.iloc[:, :target_index + 1]

# Separazione tra feature (X) e target (y)
X = data.drop(target_column, axis=1)  # Tutte le colonne fino a etdrs_20_visit esclusa
y = data[target_column]  # Target
X = X.replace(',', '.', regex=True).astype(float)

# Applica SelectKBest con la funzione f_classif
selector = SelectKBest(score_func=f_regression, k=28)  # Seleziona le 10 migliori feature
X_new = selector.fit_transform(X, y)

# Ottieni i nomi delle feature selezionate
selected_features = X.columns[selector.get_support()]

# Visualizza le feature selezionate
print("Feature selezionate:")
print(selected_features)

# Salva il nuovo dataset con le feature selezionate
selected_data = pd.DataFrame(X_new, columns=selected_features)
selected_data['fourier_coeff'] = fourier_coeff
selected_data['trend'] = trend

save_csv(selected_data, "train_dataset_trex_not_augmented.csv")

selected_data_normalized = normalizza_feature_numeriche(selected_data,"../../../feature_engineering/regressione"
                                                                      "/scaler/scaler_trex_not_augmented.pkl")
feature_names = selected_data.columns.tolist()  # Lista delle colonne originali

joblib.dump(feature_names, "../../../feature_engineering/regressione/scaler/feature_names_trex_not_augmented.pkl")

save_csv(selected_data_normalized,'train_dataset_trex_not_augmented_normalized.csv')
