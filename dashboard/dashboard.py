import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Theme dan Style
sns.set_theme(style="whitegrid")

# Memuat kedua file
day_df = pd.read_csv("data/day.csv")
hour_df = pd.read_csv("data/hour.csv")

# Menyimpan dataset harian sebagai main_data.csv
day_df.to_csv("dashboard/main_data.csv", index=False)

# Sidebar untuk filter interaktif
st.sidebar.title("Filter Data")
selected_year = st.sidebar.selectbox("Pilih Tahun:", day_df['yr'].unique())
selected_season = st.sidebar.multiselect("Pilih Musim:", day_df['season'].unique(), default=day_df['season'].unique())

# Judul Dashboard
st.title("📊 Dashboard Analisis Bike Sharing 🚲")
st.markdown("""
    Dashboard ini menyajikan analisis mengenai data penyewaan sepeda berdasarkan beberapa fitur seperti cuaca, waktu, 
    serta faktor lainnya. Gunakan filter pada sidebar untuk melihat data berdasarkan tahun atau musim tertentu.
""")

# Mengunggah dataset
@st.cache_data
def load_data():
    data = pd.read_csv("dashboard/main_data.csv")
    return data

# Memuat data
data = load_data()

# Filter berdasarkan tahun dan musim
filtered_data = data[(data['yr'] == selected_year) & (data['season'].isin(selected_season))]

# Menampilkan informasi dataset
st.header("📄 Informasi Dataset")
st.write("Dimensi dataset:", filtered_data.shape)
st.write("Data yang ditampilkan difilter berdasarkan pilihan di sidebar.")

# Menampilkan preview data
st.subheader("Preview Data")
st.dataframe(filtered_data.head())

# Visualisasi: Grafik Jumlah Penyewaan Sepeda
st.subheader("📈 Grafik Jumlah Penyewaan Sepeda per Hari")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x='dteday', y='cnt', data=filtered_data, ax=ax, color="blue", linewidth=2.5)
ax.set_title(f"Jumlah Penyewaan Sepeda pada Tahun {selected_year}", fontsize=16)
ax.set_xlabel("Tanggal", fontsize=12)
ax.set_ylabel("Jumlah Penyewaan", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig)

# Menampilkan histogram distribusi penyewaan
st.subheader("🔍 Distribusi Penyewaan Sepeda Harian")
fig, ax = plt.subplots()
sns.histplot(filtered_data['cnt'], kde=True, color='green', ax=ax)
ax.set_title("Distribusi Penyewaan Sepeda Harian", fontsize=16)
ax.set_xlabel("Jumlah Penyewaan", fontsize=12)
ax.set_ylabel("Frekuensi", fontsize=12)
st.pyplot(fig)

# Menampilkan korelasi antara fitur cuaca dan jumlah penyewaan
st.subheader("🌦️ Korelasi Fitur Cuaca dan Penyewaan")
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = filtered_data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, linewidths=0.5, fmt=".2f", cbar_kws={"shrink": .8})
ax.set_title("Korelasi antara Fitur Cuaca dan Penyewaan", fontsize=16)
st.pyplot(fig)

# Footer atau informasi tambahan
st.markdown("""
---
### Penjelasan:
- **Jumlah Penyewaan** (cnt): Jumlah total penyewaan sepeda.
- **Fitur Cuaca**: Fitur cuaca meliputi temperatur (temp, atemp), kelembapan (hum), dan kecepatan angin (windspeed).
- **Sumber Data**: Dataset dari **UCI Machine Learning Repository** untuk analisis penyewaan sepeda.
""")
