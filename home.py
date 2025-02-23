import streamlit as st
import pandas as pd

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

'''
# Tingkat Inflasi di Indonesia
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