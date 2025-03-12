from functions.utility_functions import *

datasets = ['trex', 'gila', 'monthly']

for i in datasets:
    df = read_csv(f"../../datasets/dme/regressione/augmented/{i}.csv")

    normalizza_feature_numeriche(df, f"scaler/scaler_{i}_augmented.pkl")

    save_csv(df, f"../../datasets/dme/regressione/normalized/augmented/{i}.csv",)
