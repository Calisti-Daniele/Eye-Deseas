import numpy as np
import pandas as pd
from functions.utility_functions import *
import tensorflow as tf
from keras.api.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_sequences_with_target(data, window_size, max_future_visit):
    X_seq, X_target, X_extra, Y = [], [], [], []
    for i in range(len(data) - window_size - max_future_visit):
        for target_offset in range(1, max_future_visit + 1):
            if i + window_size + target_offset < len(data):
                sequence = data[i:i + window_size, -3].reshape(window_size, 1)  # ETDRS per l'LSTM
                trend = data[i:i + window_size, -2].reshape(window_size, 1)  # Trend
                fourier_coeff = data[i:i + window_size, -1].reshape(window_size, 1)  # Fourier Coeff

                extra_features = np.concatenate([trend, fourier_coeff], axis=1)  # Unisce trend e Fourier

                target_visit = i + window_size + target_offset

                X_seq.append(sequence)  # Sequenza ETDRS
                X_target.append([target_visit])  # Numero visita target
                X_extra.append(extra_features)  # Feature aggiuntive
                Y.append(data[i + window_size + target_offset, -3])  # Predice ETDRS

    return np.array(X_seq), np.array(X_target), np.array(X_extra), np.array(Y)


df = read_csv('../../datasets/dme/regressione/normalized/augmented/trex.csv')

window_size = 10  # Numero di visite passate come input
max_future_visit = 5  # Possiamo prevedere fino a 5 visite in avanti

X_seq, X_target, X_extra, Y = create_sequences_with_target(df.values, window_size, max_future_visit)

# Controllo della forma prima del training
print("Forma di X_seq:", X_seq.shape)  # (batch_size, window_size, 1)
print("Forma di X_target:", X_target.shape)  # (batch_size, 1)
print("Forma di X_extra:", X_extra.shape)  # (batch_size, window_size, 2)
print("Forma di Y:", Y.shape)  # (batch_size,)


# Definizione degli input
input_sequence = Input(shape=(window_size, 1))  # Sequenza di ETDRS
input_target = Input(shape=(1,))  # Numero della visita target
input_extra = Input(shape=(window_size, 2))  # Trend e Fourier Coeff

# Struttura LSTM con più unità
lstm_layer = LSTM(100, activation='relu', return_sequences=True)(input_sequence)
lstm_layer = Dropout(0.3)(lstm_layer)  # Dropout per evitare overfitting
lstm_layer = LSTM(100, activation='relu', return_sequences=False)(lstm_layer)
lstm_layer = Dropout(0.3)(lstm_layer)  # Secondo Dropout

# Elaborazione delle feature aggiuntive
extra_layer = LSTM(50, activation='relu', return_sequences=False)(input_extra)

# Concatenazione di LSTM + visita target + feature extra
merged = Concatenate()([lstm_layer, input_target, extra_layer])
dense_layer = Dense(50, activation='relu')(merged)
dense_layer = Dropout(0.2)(dense_layer)  # Dropout nel livello denso
output = Dense(1)(dense_layer)  # Output: valore ETDRS predetto

# Creazione del modello
model = Model(inputs=[input_sequence, input_target, input_extra], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Divisione in training e test set
X_train_seq, X_test_seq, X_train_target, X_test_target, X_train_extra, X_test_extra, Y_train, Y_test = train_test_split(
    X_seq, X_target, X_extra, Y, test_size=0.2, random_state=42
)

# Addestramento del modello
history = model.fit(
    [X_train_seq, X_train_target, X_train_extra],
    Y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=True
)

# Predizione
Y_pred = model.predict([X_test_seq, X_test_target, X_test_extra])

# Grafico di confronto tra valori reali e predetti
plt.figure(figsize=(10, 5))
plt.plot(Y_test, label="Valori Reali")
plt.plot(Y_pred, label="Predizioni", linestyle="dashed")
plt.legend()
plt.title("Confronto tra Valori Reali e Predetti")
plt.show()
