import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="🌬️ EcoAir Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
        animation: fadeIn 1.5s ease-out;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInUp 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        animation: slideInLeft 0.6s ease-out;
    }
    
    .good-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        animation: slideInLeft 1s ease-out;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        animation: slideInLeft 1.2s ease-out;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stTab [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTab [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    
    .stTab [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        color: white;
        animation: slideInUp 1s ease-out;
    }
    
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 3rem;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    .stMetric label {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    .floating-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

# Load data (simulasi - ganti dengan path file Anda)
@st.cache_data
def load_data():
    return pd.read_csv("FinalProject_BigData/Air_Quality.csv")

df = load_data()

model = joblib.load('FinalProject_BIgData/model_aqi.pkl')

# Konversi kolom tanggal
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M").astype(str)

# Header utama dengan animasi
st.markdown('<h1 class="main-header">🌱 EcoAir Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Dashboard Kualitas Udara Global dengan AI-Powered Insights</p>', unsafe_allow_html=True)

# Sidebar yang lebih menarik
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🔍 Smart Insights</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter interaktif
    selected_cities = st.multiselect(
        "🏙️ Pilih Kota",
        options=df['City'].unique(),
        default=df['City'].unique()[:3],
        help="Pilih kota untuk analisis"
    )
    
    date_range = st.date_input(
        "📅 Rentang Tanggal",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )
    
    # Filter data
    if selected_cities:
        filtered_df = df[
            (df['City'].isin(selected_cities)) &
            (df['Date'].dt.date >= date_range[0]) &
            (df['Date'].dt.date <= date_range[1])
        ]
    else:
        filtered_df = df
    
    st.markdown("---")
    
    # AI-powered insights
    if not filtered_df.empty:
        worst_city_aqi = filtered_df.loc[filtered_df['AQI'].idxmax()]
        best_city_aqi = filtered_df.loc[filtered_df['AQI'].idxmin()]
        
        st.markdown("""
        <div class="good-card">
            <h4>🌟 Kota Terbaik</h4>
            <p><strong>{}</strong><br>AQI: {:.1f}</p>
        </div>
        """.format(best_city_aqi['City'], best_city_aqi['AQI']), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="danger-card">
            <h4>⚠️ Perlu Perhatian</h4>
            <p><strong>{}</strong><br>AQI: {:.1f}</p>
        </div>
        """.format(worst_city_aqi['City'], worst_city_aqi['AQI']), unsafe_allow_html=True)
        
        # Trend analysis
        recent_avg = filtered_df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        
        st.markdown("### 📊 Ranking Kota")
        for i, (city, aqi) in enumerate(recent_avg.head(3).items()):
            if i == 0:
                emoji = "🥇"
                color = "#FFD700"
            elif i == 1:
                emoji = "🥈"
                color = "#C0C0C0"
            else:
                emoji = "🥉"
                color = "#CD7F32"
            
            st.markdown(f"""
            <div style="background: {color}; padding: 0.5rem; border-radius: 8px; margin: 0.2rem 0; color: white;">
                {emoji} <strong>{city}</strong>: {aqi:.1f}
            </div>
            """, unsafe_allow_html=True)

# Real-time style metrics
st.markdown("## 📈 Live Monitoring Dashboard")

if not filtered_df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_aqi = filtered_df['AQI'].mean()
        delta_aqi = avg_aqi - df['AQI'].mean()
        st.metric(
            "🌬️ Rata-rata AQI", 
            f"{avg_aqi:.1f}",
            delta=f"{delta_aqi:+.1f}",
            help="Air Quality Index"
        )
    
    with col2:
        avg_pm25 = filtered_df['PM2.5'].mean()
        delta_pm25 = avg_pm25 - df['PM2.5'].mean()
        st.metric(
            "🔴 PM2.5", 
            f"{avg_pm25:.1f}",
            delta=f"{delta_pm25:+.1f}",
            help="Particulate Matter 2.5"
        )
    
    with col3:
        avg_pm10 = filtered_df['PM10'].mean()
        delta_pm10 = avg_pm10 - df['PM10'].mean()
        st.metric(
            "🟤 PM10", 
            f"{avg_pm10:.1f}",
            delta=f"{delta_pm10:+.1f}",
            help="Particulate Matter 10"
        )
    
    with col4:
        unhealthy_days = len(filtered_df[filtered_df['AQI'] > 100])
        total_days = len(filtered_df)
        unhealthy_pct = (unhealthy_days / total_days) * 100 if total_days > 0 else 0
        st.metric(
            "⚠️ Hari Tidak Sehat", 
            f"{unhealthy_pct:.1f}%",
            help="Persentase hari dengan AQI > 100"
        )
    
    with col5:
        cities_count = len(selected_cities)
        st.metric(
            "🏙️ Kota Dipantau", 
            f"{cities_count}",
            help="Jumlah kota dalam analisis"
        )

# Enhanced Tabs
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Interactive Charts", 
    "🗺️ Geo Analytics", 
    "🔥 Correlation Matrix", 
    "🤖 AI Predictions",
    "📋 Smart Data Explorer"
])

with tab1:
    st.markdown("### 🎯 Interactive Pollution Monitoring")
    
    # Selector untuk jenis chart
    chart_type = st.selectbox(
        "Pilih Jenis Visualisasi",
        ["Line Chart", "Area Chart", "Bar Chart", "Scatter Plot"]
    )
    
    pollutant = st.selectbox(
        "Pilih Polutan",
        ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    )
    
    # Buat chart berdasarkan pilihan
    monthly_data = filtered_df.groupby(['Month', 'City'])[pollutant].mean().reset_index()
    
    if chart_type == "Line Chart":
        fig = px.line(
            monthly_data, 
            x='Month', 
            y=pollutant, 
            color='City',
            title=f"Trend {pollutant} per Bulan",
            markers=True
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    elif chart_type == "Area Chart":
        fig = px.area(
            monthly_data, 
            x='Month', 
            y=pollutant, 
            color='City',
            title=f"Area Chart {pollutant} per Bulan"
        )
    elif chart_type == "Bar Chart":
        fig = px.bar(
            monthly_data, 
            x='Month', 
            y=pollutant, 
            color='City',
            title=f"Perbandingan {pollutant} per Bulan",
            barmode='group'
        )
    else:  # Scatter Plot
        fig = px.scatter(
            filtered_df, 
            x='PM2.5', 
            y='PM10', 
            color='City',
            size='AQI',
            title="Korelasi PM2.5 vs PM10 (Ukuran = AQI)",
            hover_data=['Date']
        )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        font=dict(size=12),
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🌍 Geographic Air Quality Analysis")
    
    # Simulasi koordinat untuk kota
    city_coords = {
        'Brasilia': [-15.7942, -47.8825],
        'Cairo': [30.0444, 31.2357],
        'Dubai': [25.276987, 55.296249],
        'London': [51.5074, -0.1278],
        'New York': [40.7128, -74.0060],
        'Sydney': [-33.8688, 151.2093]
    }
    
    # Buat data untuk map
    city_summary = filtered_df.groupby('City').agg({
        'AQI': 'mean',
        'PM2.5': 'mean',
        'PM10': 'mean'
    }).reset_index()
    
    city_summary['lat'] = city_summary['City'].map(lambda x: city_coords.get(x, [0, 0])[0])
    city_summary['lon'] = city_summary['City'].map(lambda x: city_coords.get(x, [0, 0])[1])
    
    # Scatter mapbox
    fig_map = px.scatter_mapbox(
        city_summary,
        lat='lat',
        lon='lon',
        size='AQI',
        color='PM2.5',
        hover_name='City',
        hover_data=['AQI', 'PM2.5', 'PM10'],
        color_continuous_scale='Viridis',
        size_max=50,
        zoom=1,
        title="Global Air Quality Map"
    )
    
    fig_map.update_layout(
        mapbox_style="open-street-map",
        height=600,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Bubble chart untuk comparison
    fig_bubble = px.scatter(
        city_summary,
        x='PM2.5',
        y='PM10',
        size='AQI',
        color='City',
        title="Bubble Chart: PM2.5 vs PM10 (Size = AQI)",
        labels={'PM2.5': 'PM2.5 (μg/m³)', 'PM10': 'PM10 (μg/m³)'}
    )
    
    fig_bubble.update_layout(height=400)
    st.plotly_chart(fig_bubble, use_container_width=True)

with tab3:
    st.markdown("### 🔥 Advanced Correlation Analysis")
    
    # Heatmap korelasi
    corr_data = filtered_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].corr()
    
    fig_heatmap = px.imshow(
        corr_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Correlation Matrix of Air Pollutants"
    )
    
    fig_heatmap.update_layout(
        width=700,
        height=500,
        title_font_size=16
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Pairplot untuk analisis lebih detail
    st.markdown("#### 📊 Pairwise Relationships")
    
    selected_vars = st.multiselect(
        "Pilih variabel untuk analisis:",
        ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI'],
        default=['PM2.5', 'PM10', 'AQI']
    )
    
    if len(selected_vars) >= 2:
        fig_scatter_matrix = px.scatter_matrix(
            filtered_df,
            dimensions=selected_vars,
            color='City',
            title="Scatter Matrix of Selected Pollutants"
        )
        fig_scatter_matrix.update_layout(height=600)
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)

with tab4:
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Powered AQI Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        co_input = st.number_input("Input Nilai CO", min_value=0.0, value=250.0)
        no2_input = st.number_input("Input Nilai NO2", min_value=0.0, value=30.0)
    with col2:
        so2_input = st.number_input("Input Nilai SO2", min_value=0.0, value=10.0)
        o3_input = st.number_input("Input Nilai O3", min_value=0.0, value=50.0)
    with col3:
        pm25_input = st.number_input("Input Nilai PM2.5", min_value=0.0, value=20.0)
        pm10_input = st.number_input("Input Nilai PM10", min_value=0.0, value=40.0)
    
    if st.button("🚀 Prediksi AQI dengan AI", type="primary"):
        # Simulasi prediksi ML (ganti dengan model yang sebenarnya)
        X_input = np.array([[co_input, no2_input, so2_input, o3_input, pm25_input, pm10_input]])  # atau termasuk CO jika modelmu pakai
        predicted_aqi = model.predict(X_input)[0]
        
        # Tampilkan hasil dengan styling menarik
        col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
        
        with col_pred2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                <h2 style="color: white; margin: 0;">🎯 Prediksi AQI</h2>
                <h1 style="color: white; font-size: 3rem; margin: 0.5rem 0;">{predicted_aqi:.1f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Kategori dan rekomendasi
        if predicted_aqi <= 50:
            category = "🟢 Baik"
            color = "#4CAF50"
            recommendation = "Kualitas udara baik. Aman untuk semua aktivitas outdoor!"
        elif predicted_aqi <= 100:
            category = "🟡 Sedang"
            color = "#FF9800"
            recommendation = "Kualitas udara sedang. Kelompok sensitif sebaiknya mengurangi aktivitas outdoor."
        elif predicted_aqi <= 150:
            category = "🟠 Tidak Sehat (Sensitif)"
            color = "#FF5722"
            recommendation = "Tidak sehat untuk kelompok sensitif. Batasi aktivitas outdoor yang lama."
        elif predicted_aqi <= 200:
            category = "🔴 Tidak Sehat"
            color = "#F44336"
            recommendation = "Tidak sehat untuk semua orang. Hindari aktivitas outdoor."
        else:
            category = "🟣 Sangat Tidak Sehat"
            color = "#9C27B0"
            recommendation = "Berbahaya! Tetap di dalam ruangan dan gunakan air purifier."
        
        st.markdown(f"""
        <div style="background: {color}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; color: white;">
            <h3 style="margin: 0;">Kategori: {category}</h3>
            <p style="margin: 0.5rem 0 0 0;">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge chart untuk visualisasi
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_aqi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "AQI Meter"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 300]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#E8F5E8"},
                    {'range': [50, 100], 'color': "#FFF3E0"},
                    {'range': [100, 150], 'color': "#FFF0F0"},
                    {'range': [150, 200], 'color': "#FFEBEE"},
                    {'range': [200, 300], 'color': "#F3E5F5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 150
                }
            }
        ))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown("### 📋 Smart Data Explorer")
    
    # Filter controls
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        show_city = st.selectbox(
            "🏙️ Filter Kota:", 
            ["Semua"] + list(filtered_df['City'].unique())
        )
    
    with col_filter2:
        aqi_range = st.slider(
            "🌬️ Rentang AQI:",
            min_value=int(filtered_df['AQI'].min()),
            max_value=int(filtered_df['AQI'].max()),
            value=(int(filtered_df['AQI'].min()), int(filtered_df['AQI'].max()))
        )
    
    with col_filter3:
        show_columns = st.multiselect(
            "📊 Pilih Kolom:",
            options=['Date', 'City', 'CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI'],
            default=['Date', 'City', 'PM2.5', 'PM10', 'AQI']
        )
    
    # Apply filters
    display_df = filtered_df.copy()
    
    if show_city != "Semua":
        display_df = display_df[display_df['City'] == show_city]
    
    display_df = display_df[
        (display_df['AQI'] >= aqi_range[0]) & 
        (display_df['AQI'] <= aqi_range[1])
    ]
    
    if show_columns:
        display_df = display_df[show_columns]
    
    # Data preview with styling
    st.markdown("#### 📊 Data Preview")
    
    # Add color coding based on AQI
    def highlight_aqi(val):
        if 'AQI' in str(val):
            return ''
        try:
            val = float(val)
            if val <= 50:
                return 'background-color: #E8F5E8'
            elif val <= 100:
                return 'background-color: #FFF3E0'
            elif val <= 150:
                return 'background-color: #FFF0F0'
            elif val <= 200:
                return 'background-color: #FFEBEE'
            else:
                return 'background-color: #F3E5F5'
        except:
            return ''
    
    # Show data with pagination
    rows_per_page = st.selectbox("Baris per halaman:", [10, 25, 50, 100], index=1)
    
    if len(display_df) > rows_per_page:
        total_pages = len(display_df) // rows_per_page + (1 if len(display_df) % rows_per_page > 0 else 0)
        page = st.selectbox("Halaman:", range(1, total_pages + 1))
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        paginated_df = display_df.iloc[start_idx:end_idx]
    else:
        paginated_df = display_df
    
    # Display the data
    if 'AQI' in paginated_df.columns:
        styled_df = paginated_df.style.applymap(
            lambda x: highlight_aqi(x) if isinstance(x, (int, float)) else '',
            subset=['AQI'] if 'AQI' in paginated_df.columns else []
        )
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        st.dataframe(paginated_df, use_container_width=True, height=400)
    
    # Quick stats
    if not paginated_df.empty:
        st.markdown("#### 📈 Quick Statistics")
        
        numeric_cols = paginated_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.markdown("**📊 Descriptive Statistics**")
                st.dataframe(paginated_df[numeric_cols].describe().round(2))
            
            with col_stats2:
                st.markdown("**🎯 Key Metrics**")
                for col in numeric_cols[:5]:  # Show first 5 numeric columns
                    avg_val = paginated_df[col].mean()
                    max_val = paginated_df[col].max()
                    min_val = paginated_df[col].min()
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); 
                                padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; color: white;">
                        <strong>{col}:</strong> Avg: {avg_val:.1f} | Max: {max_val:.1f} | Min: {min_val:.1f}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Download section
    st.markdown("#### 💾 Download Data")
    col_download1, col_download2, col_download3 = st.columns(3)
    
    with col_download1:
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"air_quality_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col_download2:
        json_data = display_df.to_json(orient='records', date_format='iso')
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"air_quality_data_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col_download3:
        # Summary report
        summary_report = f"""
        AIR QUALITY REPORT
        ==================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Dataset Summary:
        - Total Records: {len(display_df)}
        - Cities Analyzed: {', '.join(display_df['City'].unique()) if 'City' in display_df.columns else 'N/A'}
        - Date Range: {display_df['Date'].min()} to {display_df['Date'].max() if 'Date' in display_df.columns else 'N/A'}
        
        Key Insights:
        - Average AQI: avg_aqi = f"{display_df['AQI'].mean():.1f}" if 'AQI' in display_df.columns else "N/A"st.markdown(f"- Average AQI: {avg_aqi}")
        - Highest AQI: high_aqi = f"{display_df['AQI'].max():.1f}" if 'AQI' in display_df.columns else "N/A"st.markdown(f"- Average AQI: {avg_aqi}")
        - Days with Unhealthy Air (AQI > 100): {len(display_df[display_df['AQI'] > 100]) if 'AQI' in display_df.columns else 'N/A'}
        """
        
        st.download_button(
            label="📥 Download Report",
            data=summary_report,
            file_name=f"air_quality_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# Real-time alerts section
st.markdown("---")
st.markdown("## 🚨 Smart Alert System")

alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    if not filtered_df.empty:
        critical_cities = filtered_df[filtered_df['AQI'] > 150]['City'].unique()
        if len(critical_cities) > 0:
            st.markdown("""
            <div class="danger-card">
                <h4>🚨 ALERT KRITIS</h4>
                <p><strong>Kota dengan AQI > 150:</strong><br>
                {}</p>
                <p><em>Segera ambil tindakan protektif!</em></p>
            </div>
            """.format(', '.join(critical_cities)), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="good-card">
                <h4>✅ STATUS AMAN</h4>
                <p>Tidak ada kota dengan AQI kritis saat ini</p>
            </div>
            """, unsafe_allow_html=True)

with alert_col2:
    if not filtered_df.empty:
        pm25_alert = filtered_df[filtered_df['PM2.5'] > 35]['City'].unique()
        if len(pm25_alert) > 0:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ PERINGATAN PM2.5</h4>
                <p><strong>Kota dengan PM2.5 > 35:</strong><br>
                {}</p>
                <p><em>Gunakan masker saat keluar!</em></p>
            </div>
            """.format(', '.join(pm25_alert)), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="good-card">
                <h4>😷 PM2.5 TERKENDALI</h4>
                <p>Semua kota dalam batas aman PM2.5</p>
            </div>
            """, unsafe_allow_html=True)

with alert_col3:
    if not filtered_df.empty:
        trend_data = filtered_df.groupby('Date')['AQI'].mean().tail(7)
        if len(trend_data) >= 2:
            trend_change = trend_data.iloc[-1] - trend_data.iloc[0]
            if trend_change > 10:
                trend_status = "📈 MENINGKAT"
                trend_color = "danger-card"
            elif trend_change < -10:
                trend_status = "📉 MEMBAIK"
                trend_color = "good-card"
            else:
                trend_status = "➡️ STABIL"
                trend_color = "insight-card"
            
            st.markdown(f"""
            <div class="{trend_color}">
                <h4>📊 TREND 7 HARI</h4>
                <p><strong>{trend_status}</strong><br>
                Perubahan: {trend_change:+.1f} poin</p>
            </div>
            """, unsafe_allow_html=True)

# Advanced Analytics Section
st.markdown("---")
st.markdown("## 🔬 Advanced Analytics")

advanced_col1, advanced_col2 = st.columns(2)

with advanced_col1:
    st.markdown("### 📊 Pollution Index Radar")
    
    if not filtered_df.empty and len(selected_cities) > 0:
        # Create radar chart
        categories = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
        
        fig_radar = go.Figure()
        
        for city in selected_cities[:6]:  # Limit to 6 cities for readability
            city_data = filtered_df[filtered_df['City'] == city]
            if not city_data.empty:
                values = [city_data[cat].mean() for cat in categories]
                # Normalize values to 0-100 scale for better visualization
                max_vals = [filtered_df[cat].max() for cat in categories]
                normalized_values = [(val/max_val)*100 for val, max_val in zip(values, max_vals)]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=categories,
                    fill='toself',
                    name=city
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Pollution Profile Comparison",
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

with advanced_col2:
    st.markdown("### 🎯 AQI Distribution")
    
    if not filtered_df.empty:
        # Create AQI distribution histogram
        fig_hist = px.histogram(
            filtered_df,
            x='AQI',
            color='City',
            title='AQI Distribution by City',
            nbins=30,
            opacity=0.7
        )
        
        # Add vertical lines for AQI categories
        fig_hist.add_vline(x=50, line_dash="dash", line_color="green", 
                          annotation_text="Good")
        fig_hist.add_vline(x=100, line_dash="dash", line_color="yellow", 
                          annotation_text="Moderate")
        fig_hist.add_vline(x=150, line_dash="dash", line_color="orange", 
                          annotation_text="Unhealthy")
        fig_hist.add_vline(x=200, line_dash="dash", line_color="red", 
                          annotation_text="Very Unhealthy")
        
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

# Footer dengan informasi dan links
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🌍 EcoAir Analytics Dashboard</h3>
    <p>Powered by Advanced Data Science & Machine Learning</p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
        <div>🤖 AI Predictions</div>
        <div>🌱 Environmental Impact</div>
        <div>📱 Mobile Responsive</div>
    </div>
    <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        Data Sources: <a href='https://www.kaggle.com/datasets/youssefelebiary/global-air-quality-2023-6-cities'>[Kaggle - Global Air Quality 2023]</a>  | 
        Built with Streamlit & Plotly | 
        © 2024 EcoAir Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Add some JavaScript for extra interactivity (optional)
st.markdown("""
<script>
// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Add floating animation to metric cards
const cards = document.querySelectorAll('.metric-card');
cards.forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-5px) scale(1.02)';
    });
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});
</script>
""", unsafe_allow_html=True)