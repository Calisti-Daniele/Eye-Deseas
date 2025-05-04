import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from functions.utility_functions import read_csv

warnings.filterwarnings("ignore")

# Carica il dataset unificato
df = read_csv('../../datasets/dme/classificazione/dataset_unificato_senza visite.csv')

# Separa le feature e il target
X = df.drop(columns=['trattamento'])
X = X.select_dtypes(include=['number'])  # Mantieni solo colonne numeriche
y = df['trattamento']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inizializza e allena CatBoost
model = CatBoostClassifier(verbose=True, random_state=42)
model.fit(X_train, y_train)

# Predizione
y_pred = model.predict(X_test)

# Report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matrice di Confusione - CatBoost')
plt.tight_layout()
plt.show()

# Salvataggio modello
with open('catboost_model_no_visits.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n💾 Modello CatBoost salvato in 'catboost_model.pkl'.")