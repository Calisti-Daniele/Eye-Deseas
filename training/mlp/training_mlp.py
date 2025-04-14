import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.callbacks import EarlyStopping
from keras.api.models import load_model
import joblib
from functions.utility_functions import *
from tensorflow.keras.losses import MeanSquaredError

# Caricamento e preparazione del dataset
data = read_csv('../xgboost/feature_selection/train_dataset_trex_not_augmented_normalized.csv')
data = data.map(convert_to_float)

# Selezione delle feature
target_column = "etdrs_19_visit"
target_index = data.columns.get_loc(target_column)
selected_features = list(data.columns[:target_index + 1]) + ["etdrs_20_visit", "fourier_coeff", "trend"]
data = data[selected_features]

X = data.drop("etdrs_20_visit", axis=1)
y = data["etdrs_20_visit"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Definizione del modello
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer per regressione
])

model.compile(optimizer='adam', loss=MeanSquaredError())

# Early stopping per evitare overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Valutazione
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE sul test set: {rmse:.4f}")

# Salvataggio del modello
model_save_path = "../../models/best_model_not_augmented_neural.h5"
model.save(model_save_path)
print(f"\nModello salvato in: {model_save_path}")
