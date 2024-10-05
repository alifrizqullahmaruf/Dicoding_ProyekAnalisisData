import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set Theme dan Style untuk visualisasi
sns.set_theme(style="whitegrid")

# Memuat kedua file CSV yang berisi data harian dan per jam
day_df = pd.read_csv("data/day.csv")
hour_df = pd.read_csv("data/hour.csv")

# Menyimpan dataset harian sebagai main_data.csv di folder dashboard
day_df.to_csv("dashboard/main_data.csv", index=False)

# Sidebar untuk filter interaktif
st.sidebar.title("Filter Data")
selected_year = st.sidebar.selectbox("Pilih Tahun:", day_df['yr'].unique())  # Memilih tahun dari data yang tersedia
selected_season = st.sidebar.multiselect("Pilih Musim:", day_df['season'].unique(), default=day_df['season'].unique())  # Memilih musim

# Judul Dashboard
st.title("📊 Dashboard Analisis Bike Sharing 🚲")
st.markdown("""
    Dashboard ini menyajikan analisis mengenai data penyewaan sepeda berdasarkan beberapa fitur seperti cuaca, waktu, 
    serta faktor lainnya. Gunakan filter pada sidebar untuk melihat data berdasarkan tahun atau musim tertentu.
""")

# Mengunggah dataset dan menyimpan cache untuk mempercepat loading data
@st.cache_data
def load_data():
    data = pd.read_csv("dashboard/main_data.csv")  # Memuat dataset yang telah disimpan
    return data

# Memuat data dari fungsi load_data
data = load_data()

# Filter data berdasarkan tahun dan musim yang dipilih
filtered_data = data[(data['yr'] == selected_year) & (data['season'].isin(selected_season))]

# Menampilkan informasi tentang dataset yang difilter
st.header("📄 Informasi Dataset")
st.write("Dimensi dataset:", filtered_data.shape)  # Menampilkan dimensi data
st.write("Data yang ditampilkan difilter berdasarkan pilihan di sidebar.")

# Menampilkan preview dari data yang difilter
st.subheader("Preview Data")
st.dataframe(filtered_data.head())  # Menampilkan 5 baris pertama dari dataset yang difilter

# Visualisasi 1: Faktor yang Mempengaruhi Jumlah Penyewaan Sepeda Harian
st.subheader("📊 Faktor yang Mempengaruhi Jumlah Penyewaan Sepeda Harian")

# Korelasi antara fitur cuaca dan jumlah penyewaan
st.write("Berikut ini adalah korelasi antara beberapa faktor cuaca dengan jumlah penyewaan sepeda:")

# Membuat heatmap untuk visualisasi korelasi
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = filtered_data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()  # Menghitung korelasi
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, linewidths=0.5, fmt=".2f", cbar_kws={"shrink": .8})  # Membuat heatmap
ax.set_title("Korelasi antara Fitur Cuaca dan Jumlah Penyewaan", fontsize=16)
st.pyplot(fig)  # Menampilkan heatmap di dashboard

st.markdown("""
**Analisis Korelasi:**  
Heatmap di atas menunjukkan korelasi antara beberapa fitur cuaca (temperatur, kelembapan, dan kecepatan angin) 
dengan jumlah penyewaan sepeda (cnt). Angka yang mendekati 1 menunjukkan hubungan positif yang kuat, sementara angka 
yang mendekati -1 menunjukkan hubungan negatif. Dari heatmap ini, kita dapat melihat bahwa temperatur memiliki 
korelasi positif yang signifikan dengan jumlah penyewaan, sementara kelembapan dan kecepatan angin memiliki korelasi 
yang lebih rendah.
""")

# Model Regresi Linier untuk melihat pengaruh fitur terhadap jumlah penyewaan
X = filtered_data[['temp', 'hum', 'windspeed']]  # Fitur independen
y = filtered_data['cnt']  # Fitur dependen (jumlah penyewaan)

# Melatih model regresi linier
model = LinearRegression()
model.fit(X, y)

# Mengambil koefisien regresi
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Koefisien'])
st.write("Koefisien dari model regresi linier yang memprediksi jumlah penyewaan sepeda berdasarkan fitur cuaca:")
st.dataframe(coefficients)  # Menampilkan koefisien dalam tabel

st.markdown("""
**Model Regresi Linier:**  
Model regresi linier ini membantu kita untuk memahami seberapa besar pengaruh setiap faktor (temperatur, 
kelembapan, dan kecepatan angin) terhadap jumlah penyewaan sepeda. Koefisien yang lebih tinggi menunjukkan pengaruh 
yang lebih besar. Misalnya, jika koefisien temperatur positif dan signifikan, maka peningkatan temperatur 
berpotensi meningkatkan jumlah penyewaan sepeda.
""")

# Visualisasi 2: Pola Penggunaan Kasual vs Terdaftar berdasarkan Waktu dan Cuaca
st.subheader("⏳ Pola Penggunaan Sepeda oleh Pengguna Kasual dan Terdaftar")

# Visualisasi pola penggunaan berdasarkan waktu (contoh: bulanan)
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x='dteday', y='casual', data=filtered_data, label='Pengguna Kasual', ax=ax, color="orange")  # Menggambarkan data pengguna kasual
sns.lineplot(x='dteday', y='registered', data=filtered_data, label='Pengguna Terdaftar', ax=ax, color="blue")  # Menggambarkan data pengguna terdaftar
ax.set_title(f"Pola Penggunaan Sepeda oleh Pengguna Kasual dan Terdaftar pada Tahun {selected_year}", fontsize=16)
ax.set_xlabel("Tanggal", fontsize=12)
ax.set_ylabel("Jumlah Pengguna", fontsize=12)
plt.xticks(rotation=45)  # Memutar label sumbu x
st.pyplot(fig)  # Menampilkan grafik di dashboard

st.markdown("""
**Pola Penggunaan Kasual vs Terdaftar:**  
Grafik di atas menunjukkan pola penggunaan sepeda oleh pengguna kasual dan terdaftar dari waktu ke waktu. 
Pengguna kasual (garis oranye) biasanya memiliki pola yang lebih fluktuatif dibandingkan pengguna terdaftar (garis biru), 
yang menunjukkan pola penggunaan yang lebih stabil. Hal ini dapat memberikan wawasan tentang kapan waktu puncak 
penggunaan sepeda terjadi dan membantu dalam perencanaan operasional.
""")

# Distribusi penggunaan sepeda oleh pengguna kasual dan terdaftar berdasarkan cuaca
st.subheader("🌤️ Distribusi Penggunaan Sepeda Berdasarkan Cuaca")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='temp', y='casual', data=filtered_data, label='Pengguna Kasual', ax=ax, color="orange")  # Menggambarkan data pengguna kasual
sns.scatterplot(x='temp', y='registered', data=filtered_data, label='Pengguna Terdaftar', ax=ax, color="blue")  # Menggambarkan data pengguna terdaftar
ax.set_title(f"Penggunaan Sepeda Berdasarkan Suhu pada Tahun {selected_year}", fontsize=16)
ax.set_xlabel("Suhu (temp)", fontsize=12)
ax.set_ylabel("Jumlah Pengguna", fontsize=12)
st.pyplot(fig)  # Menampilkan grafik di dashboard

st.markdown("""
**Distribusi Penggunaan Berdasarkan Suhu:**  
Scatter plot ini menunjukkan hubungan antara suhu dan jumlah pengguna sepeda. Pengguna kasual cenderung 
menyewa lebih banyak sepeda pada suhu yang lebih tinggi, sementara pengguna terdaftar menunjukkan 
pola penggunaan yang lebih stabil terlepas dari suhu. Hal ini dapat membantu untuk memahami bagaimana 
cuaca mempengaruhi preferensi pengguna dalam menyewa sepeda.
""")

# Footer atau informasi tambahan
st.markdown("""
---
### Penjelasan:
- **Jumlah Penyewaan** (cnt): Jumlah total penyewaan sepeda.
- **Pengguna Kasual** (casual): Pengguna yang tidak terdaftar.
- **Pengguna Terdaftar** (registered): Pengguna yang sudah terdaftar dalam sistem.
- **Fitur Cuaca**: Fitur cuaca meliputi temperatur (temp), kelembapan (hum), dan kecepatan angin (windspeed).
- **Sumber Data**: Dataset dari **UCI Machine Learning Repository** untuk analisis penyewaan sepeda.
""")
