import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Masking, Input
from keras.api.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import time  # 👈 aggiunto

# === Parametri
dataset_path = "datasets/gru_dataset_random_10k.npz"
model_path = "models/gru_scheduled_sampling_bidirectional_10k.keras"
scaler_path = "scaler/scaler_gru_y_10k.pkl"
epochs = 50
batch_size = 32
scheduled_sampling_prob = 0.5

# === Caricamento dataset
print(f"📥 Caricamento dataset da {dataset_path}")
data = np.load(dataset_path)
X, y = data["X"], data["y"]
timesteps = X.shape[1]
feature_dim = X.shape[2]
print(f"✅ Dataset shape: X={X.shape}, y={y.shape}, timesteps={timesteps}, features={feature_dim}")

# === Normalizzazione target
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
os.makedirs("scaler", exist_ok=True)
joblib.dump(scaler_y, scaler_path)
print(f"🔧 Scaler salvato in {scaler_path}")

# === Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# === Costruzione modello
inputs = Input(shape=(None, feature_dim))
x = Masking(mask_value=0.0)(inputs)
x = GRU(128, return_sequences=True)(x)
x = GRU(64)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1)(x)
model = Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()
train_mae = tf.keras.metrics.MeanAbsoluteError()
val_mae = tf.keras.metrics.MeanAbsoluteError()

# === Scheduled Sampling: training step
def train_step(x_batch, y_batch, sampling_prob):
    with tf.GradientTape() as tape:
        batch_size = tf.shape(x_batch)[0]
        x_mod = tf.identity(x_batch)

        for t in range(1, timesteps):
            use_model_pred = tf.random.uniform([batch_size], 0, 1) < sampling_prob
            real_inputs = tf.squeeze(x_batch[:, t:t+1, 0], axis=-1)
            pred_inputs = tf.squeeze(model(x_mod[:, :t]), axis=-1)
            chosen_inputs = tf.where(use_model_pred, pred_inputs, real_inputs)
            chosen_inputs = tf.reshape(chosen_inputs, [-1, 1, 1])
            static_feat = tf.expand_dims(x_mod[:, t, 1:], axis=1)
            full_step = tf.concat([chosen_inputs, static_feat], axis=-1)
            x_mod = tf.concat([x_mod[:, :t], full_step, x_mod[:, t + 1:]], axis=1)

        preds = model(x_mod)
        loss = loss_fn(y_batch, preds)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_mae.update_state(y_batch, preds)
    return loss

def val_step(x_batch, y_batch):
    preds = model(x_batch)
    loss = loss_fn(y_batch, preds)
    val_mae.update_state(y_batch, preds)
    return loss

# === Training Loop
print("🚀 Inizio training con scheduled sampling reale")
start_time = time.time()  # 🕒 start timer
best_val_loss = float("inf")
os.makedirs("models", exist_ok=True)

for epoch in range(1, epochs + 1):
    print(f"\n📅 Epoch {epoch}/{epochs}")
    train_mae.reset_state()
    val_mae.reset_state()
    total_loss = 0
    total_val_loss = 0

    for step, (x_batch, y_batch) in enumerate(train_ds):
        print(f"\n📦 Batch {step + 1}")
        loss = train_step(x_batch, y_batch, scheduled_sampling_prob)
        total_loss += loss.numpy()
        print(f"📉 Batch Loss: {loss.numpy():.4f}")

    for x_batch, y_batch in test_ds:
        val_loss = val_step(x_batch, y_batch)
        total_val_loss += val_loss.numpy()

    avg_loss = total_loss / len(train_ds)
    avg_val_loss = total_val_loss / len(test_ds)

    print(f"\n📊 [Epoch {epoch}] Train Loss: {avg_loss:.4f} | MAE: {train_mae.result().numpy():.4f}")
    print(f"📊 [Epoch {epoch}] Val   Loss: {avg_val_loss:.4f} | MAE: {val_mae.result().numpy():.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save(model_path)
        print(f"✅ Miglior modello salvato in: {model_path}")

    scheduled_sampling_prob = max(0.05, scheduled_sampling_prob * 0.95)

end_time = time.time()  # 🕒 end timer
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"\n⏱️ Tempo totale di addestramento: {int(minutes)} minuti e {int(seconds)} secondi")
print("🏁 Fine training!")
