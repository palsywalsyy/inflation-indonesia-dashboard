import streamlit as st

# Set the title and favicon that appear in the Browser's tab bar.
forecast_page = st.Page("forecast.py", title="Prediksi", icon=":material/search:")
home_page = st.Page("home.py", title="Home", icon=":material/home:")
pg = st.navigation([home_page, forecast_page])

st.set_page_config(
    page_title='Prediksi',
    page_icon=':money:', # This is an emoji shortcode. Could be a URL too.
)
pg.run()
