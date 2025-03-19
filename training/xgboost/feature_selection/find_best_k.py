from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from functions.utility_functions import *

# Carica il dataset
data = read_csv('../../../datasets/dme/ready_to_use/not_augmented/trex.csv')

data.drop(['study_id'], axis=1, inplace=True)
# Trova l'indice della colonna target
target_column = "etdrs_20_visit"
target_index = data.columns.get_loc(target_column)

# Mantieni solo le colonne fino a "etdrs_20_visit"
data = data.iloc[:, :target_index + 1]

# Separazione tra feature (X) e target (y)
X = data.drop(target_column, axis=1)  # Tutte le colonne fino a etdrs_20_visit esclusa
y = data[target_column]  # Target

X = X.replace(',', '.', regex=True).astype(float)

# Prova diversi valori di k
k_values = range(1, X.shape[1] + 1)  # Da 1 al numero totale di feature
best_k = 0
best_score = float('-inf')
scores = []

print(f"num: {X.shape[1] + 1}")

for k in k_values:
    # Applica SelectKBest con k feature
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)

    # Addestra un modello di regressione lineare
    model = XGBRegressor(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        random_state=42,
        verbosity=2,
        booster='gbtree'  # Booster predefinito
    )

    # Valuta il modello con cross-validation
    # Scorer personalizzato per R²
    r2_scorer = make_scorer(lambda y_true, y_pred: r2_score(y_true, y_pred), greater_is_better=True)
    cv_score = cross_val_score(model, X_selected, y, cv=5, scoring=r2_scorer).mean()
    scores.append(cv_score)

    # Aggiorna il miglior k se necessario
    if cv_score >= best_score:
        best_score = cv_score
        best_k = k

# Stampa il miglior valore di k e il punteggio associato
print(f"Miglior valore di k: {best_k}")
print(f"Punteggio R²: {best_score}")

# Grafico delle prestazioni in funzione di k
plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, marker='o')
plt.title('Prestazioni in funzione di k (R²)')
plt.xlabel('Numero di feature selezionate (k)')
plt.ylabel('R²')
plt.grid()
plt.show()
