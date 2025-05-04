import numpy as np
from keras.src.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import Masking, GRU, Dense, Dropout, Input
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
import os
import joblib

# === Percorsi
dataset_path = "datasets/gila_dataset_random_10k.npz"
model_path = "models/gila_scheduled_10k_bidirectional.keras"
scaler_path = "scaler/scaler_gru_y_gila_10k_bidirectional.pkl"

# === Caricamento dataset
print(f"📥 Caricamento del dataset GRU da: {dataset_path}")
data = np.load(dataset_path)
X, y = data["X"], data["y"]
print(f"✅ Dataset GRU caricato: X.shape={X.shape}, y.shape={y.shape}")

# === Normalizzazione del target y
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
os.makedirs("scaler", exist_ok=True)
joblib.dump(scaler_y, scaler_path)
print(f"🔧 Scaler salvato in: {scaler_path}")

# === Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Costruzione del modello
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Masking(mask_value=0.0),
    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# === Callback
os.makedirs("models", exist_ok=True)
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True, verbose=1)
]

# === Training
print("🚀 Inizio training con sequenze randomizzate e scheduled-like input")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# === Valutazione finale
loss, mae = model.evaluate(X_test, y_test)
print(f"\n📉 Test set — MSE: {loss:.4f}, MAE: {mae:.4f}")
print(f"✅ Modello salvato in: {model_path}")
