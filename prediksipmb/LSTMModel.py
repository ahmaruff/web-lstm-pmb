import io
import base64
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


class LSTMModel:
    def __init__(self, input_shape, model_path='lstm_model.h5'):
        self.model = Sequential([
            LSTM(32, activation='tanh', input_shape=input_shape, return_sequences=True),
            Dropout(0.1),
            LSTM(16, activation='tanh'),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['accuracy']
        )

        self.model_path = model_path

        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess_data(self, raw_data, sequence_length=2):
        df = pd.DataFrame(raw_data)
        scaled_data = self.scaler.fit_transform(df['jml_mhs'].values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length, 0])
            y.append(scaled_data[i + sequence_length, 0])

        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)
        return X, y

    def train(self, raw_data, sequence_length=2, epochs=200, batch_size=1):
        X, y = self.preprocess_data(raw_data, sequence_length)
        early_stopping = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=2)
        self.model.save(self.model_path)
        return history

    def predict(self, last_sequence, num_predictions=3):
        last_sequence = self.scaler.transform(np.array(last_sequence).reshape(-1, 1))
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(num_predictions):
            next_pred = self.model.predict(current_sequence.reshape(1, len(current_sequence), 1), verbose=0)
            predictions.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred)

        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions)

    def load(self):
        self.model = load_model(self.model_path)

    def plot_predictions(self, raw_data, sequence_length=2, num_predictions=3):
        self.load()
        df = pd.DataFrame(raw_data)
        last_sequence = df['jml_mhs'].values[-sequence_length:]
        future_predictions = self.predict(last_sequence, num_predictions)

        plt.figure(figsize=(12, 6))
        plt.plot(df['tahun'], df['jml_mhs'], 'bo-', label='Data Aktual', linewidth=2)
        future_years = [df['tahun'].iloc[-1] + i for i in range(1, num_predictions + 1)]
        plt.plot(future_years, future_predictions, 'ro-', label='Prediksi LSTM', linewidth=2)
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Mahasiswa')
        plt.title('Prediksi Jumlah Mahasiswa dengan LSTM (Positive Constraint)')
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        buf.close()
        return plot_url

    def evaluate(self, raw_data, sequence_length=2):
        X, y = self.preprocess_data(raw_data, sequence_length)
        self.load()
        y_pred = self.model.predict(X)
        y_pred_real = self.scaler.inverse_transform(y_pred)
        y_real = self.scaler.inverse_transform(y.reshape(-1, 1))

        mse = mean_squared_error(y_real, y_pred_real)
        mae = mean_absolute_error(y_real, y_pred_real)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_real, y_pred_real)

        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
