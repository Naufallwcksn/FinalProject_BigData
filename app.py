import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Kualitas Udara",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("Air_Quality.csv")

# Konversi kolom tanggal jadi datetime
df["Date"] = pd.to_datetime(df["Date"])

# Tambahkan kolom bulan
df["Month"] = df["Date"].dt.to_period("M").astype(str)

# Header utama
st.markdown('<h1 class="main-header">🌬️ Dashboard Kualitas Udara Dunia 2023</h1>', 
            unsafe_allow_html=True)

# Sidebar untuk insights dan informasi
st.sidebar.title("📊 Insights & Info")
st.sidebar.markdown("---")

# Menggunakan semua data tanpa filter
filtered_df = df

# Temukan insights otomatis
worst_city_aqi = filtered_df.loc[filtered_df['AQI'].idxmax()]
best_city_aqi = filtered_df.loc[filtered_df['AQI'].idxmin()]
worst_month = filtered_df.groupby('Month')['AQI'].mean().idxmax()
best_month = filtered_df.groupby('Month')['AQI'].mean().idxmin()

# Tampilkan insights
st.sidebar.markdown("### 🔍 **Key Insights:**")
st.sidebar.success(f"🌿 **Kota Terbaik:** {best_city_aqi['City']}")
st.sidebar.error(f"⚠️ **Kota Terburuk:** {worst_city_aqi['City']}")
st.sidebar.info(f"📅 **Bulan Terbaik:** {best_month}")
st.sidebar.warning(f"📅 **Bulan Terburuk:** {worst_month}")

st.sidebar.markdown("---")

# Standar kualitas udara
st.sidebar.markdown("### 🎯 **Standar Kualitas Udara:**")
st.sidebar.markdown("""
**AQI Categories:**
- 🟢 **Baik:** 0-50
- 🟡 **Sedang:** 51-100  
- 🟠 **Tidak Sehat:** 101-150
- 🔴 **Sangat Tidak Sehat:** 151-200
- 🟣 **Berbahaya:** 201+

**PM2.5 (WHO Guidelines):**
- 🎯 **Target:** < 15 µg/m³
- ⚠️ **Batas:** < 35 µg/m³
""")

st.sidebar.markdown("---")

# Rekomendasi
st.sidebar.markdown("### 💡 **Rekomendasi:**")
avg_aqi = filtered_df['AQI'].mean()
if avg_aqi > 100:
    st.sidebar.error("🚨 Kualitas udara tidak sehat! Kurangi aktivitas outdoor.")
elif avg_aqi > 50:
    st.sidebar.warning("⚠️ Kualitas udara sedang. Waspadai aktivitas outdoor.")
else:
    st.sidebar.success("✅ Kualitas udara baik untuk aktivitas outdoor!")

# Hitung rata-rata bulanan
monthly_avg = filtered_df.groupby(['Month', 'City'])[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].mean().reset_index()
selected_cities = df['City'].unique()  # Gunakan semua kota

# Metrics overview
st.markdown("## 📊 Ringkasan Statistik")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_aqi = filtered_df['AQI'].mean()
    st.metric("Rata-rata AQI", f"{avg_aqi:.1f}")

with col2:
    avg_pm25 = filtered_df['PM2.5'].mean()
    st.metric("Rata-rata PM2.5", f"{avg_pm25:.1f} µg/m³")

with col3:
    avg_pm10 = filtered_df['PM10'].mean()
    st.metric("Rata-rata PM10", f"{avg_pm10:.1f} µg/m³")

with col4:
    max_aqi_city = filtered_df.loc[filtered_df['AQI'].idxmax(), 'City']
    st.metric("Kota AQI Tertinggi", max_aqi_city)

with col5:
    total_records = len(filtered_df)
    st.metric("Total Data", f"{total_records}")

st.markdown("---")

# Tab untuk berbagai visualisasi
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Bulanan", "📊 Perbandingan Kota", "🔥 Heatmap Korelasi", "📋 Data Table"])

with tab1:
    st.markdown("### Trend Polutan per Bulan")
    
    # Pilihan polutan untuk line chart
    pollutants = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']
    
    # Buat subplots untuk line charts
    fig_lines = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f'Trend {poll}' for poll in pollutants[:6]] + ['Trend AQI'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"colspan": 3, "secondary_y": False}, None, None]]
    )
    
    colors = px.colors.qualitative.Set1
    
    # Plot untuk setiap polutan
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1)]
    
    for i, pollutant in enumerate(pollutants):
        if i < 6:
            row, col = positions[i]
            for j, city in enumerate(selected_cities):
                city_data = monthly_avg[monthly_avg['City'] == city]
                fig_lines.add_trace(
                    go.Scatter(
                        x=city_data['Month'],
                        y=city_data[pollutant],
                        mode='lines+markers',
                        name=city,
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        else:  # AQI plot
            for j, city in enumerate(selected_cities):
                city_data = monthly_avg[monthly_avg['City'] == city]
                fig_lines.add_trace(
                    go.Scatter(
                        x=city_data['Month'],
                        y=city_data[pollutant],
                        mode='lines+markers',
                        name=city,
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=False
                    ),
                    row=3, col=1
                )
    
    fig_lines.update_layout(
        height=800,
        title_text="Trend Polutan per Bulan (2023)",
        showlegend=True
    )
    
    # Update x-axis labels
    for i in range(1, 4):
        for j in range(1, 4):
            if not (i == 3 and j > 1):
                fig_lines.update_xaxes(tickangle=45, row=i, col=j)
    
    st.plotly_chart(fig_lines, use_container_width=True)

with tab2:
    st.markdown("### Perbandingan Rata-rata Polutan per Kota")
    
    # Hitung rata-rata per kota
    city_avg = filtered_df.groupby('City')[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].mean().reset_index()
    
    # Buat bar charts
    fig_bars = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f'Rata-rata {poll}' for poll in pollutants[:6]] + ['Rata-rata AQI'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"colspan": 3, "secondary_y": False}, None, None]]
    )
    
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, pollutant in enumerate(pollutants):
        if i < 6:
            row, col = positions[i]
            fig_bars.add_trace(
                go.Bar(
                    x=city_avg['City'],
                    y=city_avg[pollutant],
                    name=pollutant,
                    marker_color=colors_bar[i],
                    showlegend=False
                ),
                row=row, col=col
            )
        else:  # AQI plot
            fig_bars.add_trace(
                go.Bar(
                    x=city_avg['City'],
                    y=city_avg[pollutant],
                    name=pollutant,
                    marker_color=colors_bar[i],
                    showlegend=False
                ),
                row=3, col=1
            )
    
    fig_bars.update_layout(
        height=800,
        title_text="Perbandingan Rata-rata Polutan per Kota (2023)",
        showlegend=False
    )
    
    # Update x-axis labels
    for i in range(1, 4):
        for j in range(1, 4):
            if not (i == 3 and j > 1):
                fig_bars.update_xaxes(tickangle=45, row=i, col=j)
    
    st.plotly_chart(fig_bars, use_container_width=True)

with tab3:
    st.markdown("### Heatmap Korelasi Antar Polutan")
    
    # Hitung korelasi
    corr_data = filtered_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].corr()
    
    # Buat heatmap dengan Plotly
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Korelasi Antar Polutan",
        width=700,
        height=500
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Interpretasi korelasi
    st.markdown("#### 🔍 Interpretasi Korelasi:")
    st.write("- **Korelasi Tinggi (> 0.7)**: Hubungan positif yang kuat")
    st.write("- **Korelasi Sedang (0.3 - 0.7)**: Hubungan positif yang moderat")
    st.write("- **Korelasi Rendah (< 0.3)**: Hubungan yang lemah")
    st.write("- **Korelasi Negatif**: Hubungan berlawanan arah")

with tab4:
    st.markdown("### 📋 Data Lengkap")
    
    # Filter untuk tampilan data
    col1, col2 = st.columns(2)
    with col1:
        show_city = st.selectbox("Pilih Kota untuk Detail:", ["Semua"] + list(df['City'].unique()))
    with col2:
        show_columns = st.multiselect(
            "Pilih Kolom:",
            options=['Date', 'City', 'Month', 'CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI'],
            default=['Date', 'City', 'CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']
        )
    
    # Filter data untuk tabel
    if show_city != "Semua":
        table_data = filtered_df[filtered_df['City'] == show_city][show_columns]
    else:
        table_data = filtered_df[show_columns]
    
    # Tampilkan tabel
    st.dataframe(
        table_data.round(2),
        use_container_width=True,
        height=400
    )
    
    # Statistik deskriptif
    st.markdown("#### 📈 Statistik Deskriptif")
    numeric_cols = [col for col in show_columns if col in ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']]
    if numeric_cols:
        st.dataframe(
            table_data[numeric_cols].describe().round(2),
            use_container_width=True
        )

# --------------------------------------------------------------------------------------------------
# ✨ SECTION: Prediksi AQI dari Input PM2.5 & PM10
# --------------------------------------------------------------------------------------------------
st.markdown("---")
st.header("🧮 Prediksi AQI Berdasarkan Input Manual")

# Form input
col_input1, col_input2, col_input3 = st.columns([1, 1, 2])

with col_input1:
    pm25_input = st.number_input("Masukkan nilai PM2.5", min_value=0.0, value=50.0, step=1.0)

with col_input2:
    pm10_input = st.number_input("Masukkan nilai PM10", min_value=0.0, value=80.0, step=1.0)

with col_input3:
    st.markdown("### ")
    if st.button("🔍 Prediksi AQI"):
        try:
            import joblib
            model = joblib.load("model_aqi.pkl")  # Pastikan model tersimpan dengan nama ini
            input_data = pd.DataFrame({
                "PM2.5": [pm25_input],
                "PM10": [pm10_input]
            })
            pred_aqi = model.predict(input_data)[0]
            st.success(f"🎯 Prediksi AQI: **{pred_aqi:.2f}**")

            # Interpretasi kategori AQI
            if pred_aqi <= 50:
                kategori = "🟢 Baik"
            elif pred_aqi <= 100:
                kategori = "🟡 Sedang"
            elif pred_aqi <= 150:
                kategori = "🟠 Tidak Sehat untuk Kelompok Sensitif"
            elif pred_aqi <= 200:
                kategori = "🔴 Tidak Sehat"
            else:
                kategori = "🟣 Sangat Tidak Sehat / Berbahaya"

            st.info(f"Kategori Kualitas Udara: **{kategori}**")

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi. Pastikan file model `model_aqi.pkl` tersedia. \n\nError: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>📊 Dashboard Kualitas Udara Dunia | 
    🌍 <a href='https://www.kaggle.com/datasets/youssefelebiary/global-air-quality-2023-6-cities'>[Kaggle - Global Air Quality 2023]</a> |
    💻 Dibuat dengan Streamlit</p>
</div>
""", unsafe_allow_html=True)