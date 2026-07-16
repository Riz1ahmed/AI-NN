import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and Prepare the Historical Stock Data
df = pd.read_csv('Stock_Price_Data_[3921].csv')
close_prices = df['Close'].values.reshape(-1, 1)

# 2. Scale the Data to Prevent Mathematical Errors
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# 3. Split the Data (70% Training, 15% Validation, 15% Testing)
train_len = int(len(scaled_prices) * 0.70)
val_len = int(len(scaled_prices) * 0.15)

train_data = scaled_prices[:train_len]
val_data = scaled_prices[train_len:train_len + val_len]
test_data = scaled_prices[train_len + val_len:]

# 4. Create Sequences (The Sliding Window Approach)
def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_val, y_val = create_sequences(val_data)
X_test, y_test = create_sequences(test_data)

# 5. Build the Long Short-Term Memory Model
def build_lstm_model():
    model = Sequential()
    model.add(Input(shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 6. Build the Gated Recurrent Unit Model
def build_gru_model():
    model = Sequential()
    model.add(Input(shape=(60, 1)))
    model.add(GRU(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm_model()
gru_model = build_gru_model()

# 7. Configure the Early Stopping Mechanism
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 8. Train the Models
print("Training the Long Short-Term Memory Model...")
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
               epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

print("Training the Gated Recurrent Unit Model...")
gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

# 9. Make Predictions on the Unseen Test Data
lstm_predictions = lstm_model.predict(X_test)
gru_predictions = gru_model.predict(X_test)

# 10. Reverse the Data Scaling
lstm_pred_real = scaler.inverse_transform(lstm_predictions)
gru_pred_real = scaler.inverse_transform(gru_predictions)
actual_real_prices = scaler.inverse_transform(y_test)

# 11. Evaluate Performance using Standard Error Metrics
def evaluate_model(actual, predicted, model_name):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print(f"--- {model_name} Performance ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}\n")

evaluate_model(actual_real_prices, lstm_pred_real, "LSTM")
evaluate_model(actual_real_prices, gru_pred_real, "GRU")