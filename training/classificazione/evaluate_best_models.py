import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from functions.utility_functions import *

warnings.filterwarnings("ignore")

# Carica il dataset unificato
df = read_csv('../../datasets/dme/classificazione/dataset_unificato_senza visite.csv')

# Separa le feature e il target
X = df.drop(columns=['trattamento'])
X = X.select_dtypes(include=['number'])  # solo colonne numeriche
y = df['trattamento']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelli da testare
models = {
    'RandomForest': RandomForestClassifier(random_state=42, verbose=1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=1),
    'LightGBM': LGBMClassifier(random_state=42, verbose=1),
    'CatBoost': CatBoostClassifier(verbose=1, random_state=42)
}

results = {}

print("\n🔍 Inizio valutazione modelli...\n")
for name, model in models.items():
    print(f"Addestramento modello: {name}...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"Accuracy media: {scores.mean():.4f}\n")

# Seleziona il miglior modello
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n🏆 Miglior modello: {best_model_name} con accuracy {results[best_model_name]:.4f}\n")

# Fit su tutto il training set e valutazione finale
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=best_model.classes_, yticklabels=best_model.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Matrice di Confusione - {best_model_name}')
plt.tight_layout()
plt.show()