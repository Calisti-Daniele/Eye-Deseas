import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import Masking, GRU, Dense, Dropout, Bidirectional, Input
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

# === Path
dataset_path = "datasets/gru_dataset_trex_padded.npz"
model_save_path = "models/gru_etdrs_model_padded.keras"
scaler_y_path = "scaler/scaler_y_gru.pkl"
scaler_X_path = "scaler/scaler_X_gru.pkl"
mask_value = 0.0

# === Load dataset
print(f"📥 Caricamento dataset da: {dataset_path}")
data = np.load(dataset_path)
X_raw, y_raw = data["X"], data["y"]
print(f"✅ Dataset caricato: X.shape={X_raw.shape}, y.shape={y_raw.shape}")

# === Reshape per normalizzare: (samples * timesteps, features)
n_samples, n_timesteps, n_features = X_raw.shape
X_reshaped = X_raw.reshape(-1, n_features)

# === Normalizzazione X (tutte le feature, incluso etdrs)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, n_timesteps, n_features)

# === Normalizzazione y
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

# === Salva gli scaler
os.makedirs("scaler", exist_ok=True)
joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)
print(f"💾 Scaler salvati in: {scaler_X_path} e {scaler_y_path}")

# === Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Modello GRU bidirezionale
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Masking(mask_value=mask_value),
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
    ModelCheckpoint(model_save_path, save_best_only=True, verbose=1)
]

# === Training
print("🚀 Inizio training...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# === Valutazione finale
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📉 Valutazione finale sul test set (normalizzato): MSE = {loss:.4f}, MAE = {mae:.4f}")
print(f"✅ Modello salvato in: {model_save_path}")
