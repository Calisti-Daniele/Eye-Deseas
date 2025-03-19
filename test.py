# Quando normalizzi il dataset, salvalo così:
import joblib
from functions.utility_functions import *

df = read_csv('datasets/dme/regressione/normalized/not_augmented/trex.csv')
feature_names = df.columns.tolist()  # Lista delle colonne originali
joblib.dump(feature_names, "feature_engineering/regressione/scaler/feature_names_trex_not_augmented.pkl")
