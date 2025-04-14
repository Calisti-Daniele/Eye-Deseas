import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from functions.utility_functions import read_csv

# === Percorsi
dataset_path = "datasets/dynamic_dataset_trex_10k.csv"
models_dir = "models/by_n/"
features_dir = "scaler/by_n/"

# === Crea cartelle se non esistono
os.makedirs(models_dir, exist_ok=True)
os.makedirs(features_dir, exist_ok=True)

# === Carica dataset completo
data = read_csv(dataset_path).map(lambda x: pd.to_numeric(x, errors='coerce'))
print(f"✅ Dataset caricato: {data.shape}")

# === Definisci RMSE personalizzato
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# === Grid dei parametri
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}

# === Ciclo su ogni n da 3 a 19
for n in range(3, 20):
    print(f"\n📦 Allenamento + GridSearch per n = {n}")

    subset = data[data["n"] == n].copy()
    if subset.empty:
        print(f"⚠️ Nessuna riga trovata per n = {n}, salto")
        continue

    X = subset.drop(columns=["target"])
    y = subset["target"]

    # === Salva feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, f"{features_dir}features_n{n}_10k.pkl")

    # === Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )

    # === GridSearchCV
    base_model = XGBRegressor(
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        booster='gbtree',
        verbosity=0
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"✅ Migliori parametri per n={n}: {best_params}")

    # === Valutazione finale
    y_pred = best_model.predict(X_test)
    rmse_score = rmse(y_test, y_pred)
    print(f"📉 RMSE finale per n={n}: {rmse_score:.4f}")

    # === Salvataggio
    model_path = f"{models_dir}xgb_n{n}_10k.pkl"
    joblib.dump(best_model, model_path)
    print(f"💾 Modello salvato in: {model_path}")
