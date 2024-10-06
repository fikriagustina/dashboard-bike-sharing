# Import library yang diperlukan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Mengatur style seaborn
sns.set(style='dark')

# Membaca dataset
days_df = pd.read_csv("day.csv")
hours_df = pd.read_csv("hour.csv")

# Mengubah kolom 'dteday' menjadi datetime
days_df['dteday'] = pd.to_datetime(days_df['dteday'])
hours_df['dteday'] = pd.to_datetime(hours_df['dteday'])

# Rentang suhu
min_temp = hours_df["temp"].min()
max_temp = hours_df["temp"].max()

with st.sidebar:
    # Menampilkan logo perusahaan
    st.image("bike.jpeg")

    # Input filter suhu
    selected_temp = st.slider("Pilih Rentang Suhu:", min_value=min_temp, max_value=max_temp, value=(min_temp, max_temp))

    # Pemilih jumlah cluster menggunakan radio button
    num_clusters = st.radio("Pilih Jumlah Kluster:", options=[1, 2, 3, 4], index=0)

# Filter data berdasarkan rentang suhu yang dipilih
filtered_hours_df = hours_df[(hours_df["temp"] >= selected_temp[0]) & (hours_df["temp"] <= selected_temp[1])]

# Mendapatkan data pengguna
total_usage = filtered_hours_df['cnt'].sum()  # Menggunakan kolom 'cnt' untuk total
total_registered = filtered_hours_df['registered'].sum()
total_casual = filtered_hours_df['casual'].sum()

# Menampilkan metrik di dashboard
st.header('Bike Sharing Dashboard')
st.subheader('Total Pengguna Berdasarkan Suhu')

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Penggunaan Sepeda", value=total_usage)

with col2:
    st.metric("Total Pengguna Terdaftar", value=total_registered)

with col3:
    st.metric("Total Pengguna Kasual", value=total_casual)

# Segmentasi berdasarkan suhu dan waktu penggunaan dalam satu hari
st.subheader("Segmentasi Sepeda berdasarkan Suhu dan Waktu Penggunaan")

# Clustering dengan K-Means
features = filtered_hours_df[['temp', 'cnt']]  # Pastikan kolom yang ada
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_hours_df['cluster'] = kmeans.fit_predict(features_scaled)

# Visualisasi clustering
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=filtered_hours_df['temp'], y=filtered_hours_df['cnt'], hue=filtered_hours_df['cluster'], palette='viridis', ax=ax)
ax.set_title('Clustering berdasarkan Suhu dan Jumlah Pengguna')
ax.set_xlabel('Suhu (temp)')
ax.set_ylabel('Jumlah Pengguna (cnt)')
st.pyplot(fig)

# Penjelasan masing-masing cluster
st.subheader("Deskripsi Cluster")
cluster_descriptions = {
    0: "Cluster 1: Pengguna aktif di suhu dingin (suhu di bawah 10°C).",
    1: "Cluster 2: Pengguna yang sangat aktif di suhu hangat (suhu antara 20-30°C).",
    2: "Cluster 3: Pengguna sesekali di suhu sedang (suhu sekitar 15-20°C).",
    3: "Cluster 4: Pengguna dengan aktivitas rendah di suhu tinggi (suhu di atas 30°C).",
}

# Menampilkan deskripsi untuk cluster yang dipilih
for cluster_id in range(num_clusters):
    if cluster_id in cluster_descriptions:
        st.write(cluster_descriptions[cluster_id])

# Pengaruh suhu terhadap total pengguna
st.subheader("Pengaruh Suhu terhadap Total Pengguna")

# Filter data untuk analisis
hourly_usage_correlation = filtered_hours_df.groupby('hr').agg({'temp': 'mean', 'cnt': 'sum'}).reset_index()

# Hitung korelasi
correlation_hour = hourly_usage_correlation['temp'].corr(hourly_usage_correlation['cnt'])
st.write(f'Korelasi suhu dan pengguna per jam: {correlation_hour:.2f}')

# Visualisasi pengaruh suhu terhadap pengguna per jam
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=hourly_usage_correlation['temp'], y=hourly_usage_correlation['cnt'], ax=ax2)
ax2.set_title('Pengaruh Suhu terhadap Penggunaan Sepeda (Per Jam)')
ax2.set_xlabel('Suhu (temp)')
ax2.set_ylabel('Jumlah Pengguna (cnt)')
st.pyplot(fig2)

# Tambahkan visualisasi clustering berdasarkan jam dan total pengguna
st.subheader("Clustering berdasarkan Jam dan Total Pengguna")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=filtered_hours_df['hr'], y=filtered_hours_df['cnt'], hue=filtered_hours_df['cluster'], palette='viridis', ax=ax3)
ax3.set_title('Clustering berdasarkan Jam dan Total Pengguna')
ax3.set_xlabel('Jam (hr)')
ax3.set_ylabel('Jumlah Pengguna (cnt)')
st.pyplot(fig3)

# Visualisasi pengaruh suhu terhadap total pengguna sepeda (per hari)
st.subheader("Pengaruh Suhu terhadap Total Penggunaan Sepeda (Per Hari)")
daily_usage_correlation = days_df.groupby('temp').agg({'cnt': 'sum'}).reset_index()  # Ganti dengan kolom yang sesuai
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=daily_usage_correlation['temp'], y=daily_usage_correlation['cnt'], ax=ax4)
ax4.set_title('Pengaruh Suhu terhadap Total Penggunaan Sepeda (Per Hari)')
ax4.set_xlabel('Suhu (temp)')
ax4.set_ylabel('Jumlah Pengguna (cnt)')
st.pyplot(fig4)