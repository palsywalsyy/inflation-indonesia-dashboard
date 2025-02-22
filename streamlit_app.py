import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Prediksi Inflasi',
    page_icon=':money:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.
df = pd.read_csv('data/data_inflasi.csv')
df['Year'] = pd.to_datetime(df['Year'], format='%b %Y')

min_year = 1979
max_year = 2024

# Months
months = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12
} 


def create_sequences(data, timesteps=5):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)

def arima_lstm_pred(data):
    data = data['Inflasi'].values
    # Step 1: Fit ARIMA model
    arima_order = (3, 1, 3)  # Adjust based on ACF/PACF analysis
    arima_model = ARIMA(data, order=arima_order)
    arima_fit = arima_model.fit()

    # ARIMA predictions
    arima_pred = arima_fit.predict(start=0, end=len(data)-1, typ='levels')
    residuals = data - arima_pred

    # Pembagian data training dan data testing (80% untuk training, 20% untuk testing)
    train_size = int(len(df) * 0.8)
    data_training = data[:train_size]
    data_testing = data[train_size:]

    # Normalize data for LSTM
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled_training = scaler.fit_transform(data_training.reshape(-1, 1))
    data_scaled_testing = scaler.fit_transform(data_testing.reshape(-1, 1))

    # Menentukan jumlah timesteps untuk LSTM
    timesteps = 5
    X_train, y_train = create_sequences(data_scaled_training, timesteps)
    X_test, y_test = create_sequences(data_scaled_testing, timesteps)

    # Membangun Model LSTM
    model = keras.models.Sequential([
        keras.layers.LSTM(50, activation='relu',  input_shape=(timesteps, 1)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    # Melatih model LSTM
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, shuffle=False, validation_data=(X_test, y_test))

    # Forecast 12 bulan ke depan
    future_steps = 1
    future_inputs = data_scaled_testing[-timesteps:]  # Ambil window terakhir dari data uji
    future_predictions_lstm = []

    for _ in range(future_steps):
        # Ubah bentuk data agar sesuai dengan input model LSTM
        future_inputs_reshaped = future_inputs.reshape(1, timesteps, 1)

        # Prediksi menggunakan model LSTM
        lstm_pred_scaled = model.predict(future_inputs_reshaped)

        # Simpan hasil prediksi
        future_predictions_lstm.append(lstm_pred_scaled[0, 0])

        # Perbarui input dengan menggeser ke belakang dan menambahkan prediksi baru
        future_inputs = np.append(future_inputs[1:], lstm_pred_scaled, axis=0)

    # Konversi hasil prediksi ke skala asli
    future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1)).flatten()

    # Forecast menggunakan model ARIMA
    future_predictions_arima = arima_fit.forecast(steps=future_steps)

    # Kombinasikan hasil prediksi ARIMA dan LSTM
    final_forecast = future_predictions_arima + future_predictions_lstm

    # Tampilkan hasil prediksi 12 bulan ke depan
    for i, pred in enumerate(final_forecast, 1):
        print(f"Bulan ke-{i}: {pred:.2f}")

    return final_forecast[0]

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# Prediksi Inflasi
'''

# Membuat layout dengan satu row
col1, col2, col3 = st.columns(3)
with col1:
    selected_start_month = st.selectbox("Pilih Start Bulan", list(months.keys()))
    selected_start_year = st.selectbox("Pilih Start Tahun", list(range(min_year, max_year + 1)))

with col2:
    selected_end_month = st.selectbox("Pilih End Bulan", list(months.keys()))
    selected_end_year = st.selectbox("Pilih End Tahun", list(range(min_year, max_year + 1)))

with col3:
    tampilkan = st.button("Tampilkan", type="primary")
    st.button("Reset", type="secondary")

if tampilkan:
    start_date = pd.Timestamp(year=selected_start_year, month=months[selected_start_month], day=1)
    end_date = pd.Timestamp(year=selected_end_year, month=months[selected_end_month], day=1)
    
    filtered_df = df[(df['Year'] >= start_date) & (df['Year'] <= end_date)]
    st.line_chart(filtered_df, x='Year', y='Inflasi')
else:
    st.line_chart(df, x='Year', y='Inflasi')



uploaded_file = st.file_uploader("Masukkan data inflasi", type="csv")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.title(arima_lstm_pred(dataframe))