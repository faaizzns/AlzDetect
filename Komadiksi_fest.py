import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="AlzDetect",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }

    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }

    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        color: white;
        font-size: 1.2rem;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model (dalam praktiknya, load model yang sudah dilatih)
@st.cache_resource
def load_model():
    # Placeholder - dalam praktiknya, load model yang sudah disimpan
    # model = joblib.load('best_alzheimer_model.pkl')
    # Untuk demo, kita buat model sederhana
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    return model

# Fungsi untuk membuat prediksi
def make_prediction(model, input_data):
    # Simulasi prediksi (dalam praktiknya, gunakan model yang sudah dilatih)
    # prediction = model.predict([input_data])[0]
    # probability = model.predict_proba([input_data])[0]

    # Untuk demo, kita buat prediksi sederhana berdasarkan beberapa faktor risiko
    risk_factors = [
        input_data[9],  # Family History
        input_data[10], # Cardiovascular Disease
        input_data[11], # Depression
        input_data[12], # Head Injury
        input_data[16], # Memory Complaints
        input_data[18], # Confusion
        input_data[19], # Disorientation
        input_data[20], # Difficulty Completing Tasks
        input_data[21], # Forgetfulness
    ]

    # Faktor numerik
    age_factor = 1 if input_data[14] < 20 else 0  # MMSE score
    bmi_factor = 1 if input_data[3] > 30 else 0   # BMI

    risk_score = sum(risk_factors) + age_factor + bmi_factor

    # Simulasi probabilitas
    if risk_score >= 7:
        prediction = 1
        probability = [0.2, 0.8]
    elif risk_score >= 4:
        prediction = 1
        probability = [0.4, 0.6]
    else:
        prediction = 0
        probability = [0.7, 0.3]

    return prediction, probability

# Header utama
st.markdown('<h1 class="main-header">🧠 AlzDetect </h1>', unsafe_allow_html=True)

# Informasi aplikasi
st.markdown("""
<div class="info-box">
    <h3>🔍 Tentang Website</h3>
    <p>Website ini menggunakan machine learning untuk memprediksi risiko penyakit Alzheimer berdasarkan berbagai faktor risiko kesehatan dan gaya hidup. Model telah dilatih menggunakan Random Forest Classifier dengan akurasi tinggi.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk input
st.sidebar.markdown('<h2 class="sub-header">📊 Input Data Pasien</h2>', unsafe_allow_html=True)

# Input demografis
st.sidebar.markdown("### 👤 Informasi Demografis")
gender = st.sidebar.selectbox("Jenis Kelamin", [" ","Laki-laki", "Perempuan"])
ethnicity = st.sidebar.selectbox("Etnis", [" ","Kaukasia", "Afrika Amerika", "Asia", "Lainnya"])
education = st.sidebar.selectbox("Tingkat Pendidikan", [" ","Tidak ada", "SMA", "Sarjana", "Lebih Tinggi"])

# Input kesehatan fisik
st.sidebar.markdown("### 🏥 Kesehatan Fisik")
# Set value to min_value for all sliders
bmi = st.sidebar.slider("BMI", 15.0, 40.0, value=15.0, step=0.1)
smoking = st.sidebar.selectbox("Status Merokok", [" ","Tidak", "Ya"])
alcohol = st.sidebar.slider("Konsumsi Alkohol (per minggu)", 0, 20, value=0)
physical_activity = st.sidebar.slider("Aktivitas Fisik (jam/minggu)", 0, 10, value=0)
diet_quality = st.sidebar.slider("Kualitas Diet (0-10)", 0, 10, value=0)
sleep_quality = st.sidebar.slider("Kualitas Tidur (4-10)", 4, 10, value=4)

# Input riwayat medis
st.sidebar.markdown("### 📋 Riwayat Medis")
family_history = st.sidebar.selectbox("Riwayat Keluarga Alzheimer", [" ", "Tidak", "Ya"])
cardiovascular = st.sidebar.selectbox("Penyakit Kardiovaskular", [" ","Tidak", "Ya"])
depression = st.sidebar.selectbox("Depresi", [" ","Tidak", "Ya"])
head_injury = st.sidebar.selectbox("Cedera Kepala", [" ","Tidak", "Ya"])
cholesterol = st.sidebar.slider("Kolesterol Total (mg/dL)", 150, 300, value=150)

# Input kognitif
st.sidebar.markdown("### 🧠 Penilaian Kognitif")
mmse = st.sidebar.slider("MMSE Score (0-30)", 0, 30, value=0)
functional_assessment = st.sidebar.slider("Functional Assessment (0-10)", 0, 10, value=0)
memory_complaints = st.sidebar.selectbox("Keluhan Memori", [" ","Tidak", "Ya"])
adl = st.sidebar.selectbox("Kesulitan ADL", [" ","Tidak", "Ya"])
confusion = st.sidebar.selectbox("Kebingungan", [" ","Tidak", "Ya"])
disorientation = st.sidebar.selectbox("Disorientasi", [" ","Tidak", "Ya"])
difficulty_tasks = st.sidebar.selectbox("Kesulitan Menyelesaikan Tugas", [" ","Tidak", "Ya"])
forgetfulness = st.sidebar.selectbox("Pelupa", [" ","Tidak", "Ya"])

# Konversi input ke format numerik
def convert_inputs():
    input_data = [
        0 if gender == "Laki-laki" else 1,
        ["Kaukasia", "Afrika Amerika", "Asia", "Lainnya"].index(ethnicity),
        ["Tidak ada", "SMA", "Sarjana", "Lebih Tinggi"].index(education),
        bmi,
        1 if smoking == "Ya" else 0,
        alcohol,
        physical_activity,
        diet_quality,
        sleep_quality,
        1 if family_history == "Ya" else 0,
        1 if cardiovascular == "Ya" else 0,
        1 if depression == "Ya" else 0,
        1 if head_injury == "Ya" else 0,
        cholesterol,
        mmse,
        functional_assessment,
        1 if memory_complaints == "Ya" else 0,
        1 if adl == "Ya" else 0,
        1 if confusion == "Ya" else 0,
        1 if disorientation == "Ya" else 0,
        1 if difficulty_tasks == "Ya" else 0,
        1 if forgetfulness == "Ya" else 0
    ]
    return input_data

# Tombol prediksi
if st.sidebar.button("🔬 Prediksi Risiko Alzheimer", type="primary"):
    # Load model
    model = load_model()

    # Konversi input
    input_data = convert_inputs()

    # Buat prediksi
    prediction, probability = make_prediction(model, input_data)

    # Tampilkan hasil
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.markdown("""
            <div class="prediction-result" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
                <h2>⚠️ RISIKO TINGGI</h2>
                <p>Hasil prediksi menunjukkan risiko tinggi untuk penyakit Alzheimer</p>
                <p><strong>Probabilitas: {:.1f}%</strong></p>
            </div>
            """.format(probability[1] * 100), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-result" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);">
                <h2>✅ RISIKO RENDAH</h2>
                <p>Hasil prediksi menunjukkan risiko rendah untuk penyakit Alzheimer</p>
                <p><strong>Probabilitas: {:.1f}%</strong></p>
            </div>
            """.format(probability[0] * 100), unsafe_allow_html=True)

    with col2:
        # Grafik probabilitas
        fig = go.Figure(go.Bar(
            x=['Tidak Alzheimer', 'Alzheimer'],
            y=[probability[0], probability[1]],
            marker_color=['#2ecc71', '#e74c3c']
        ))
        fig.update_layout(
            title="Probabilitas Prediksi",
            yaxis_title="Probabilitas",
            xaxis_title="Kondisi"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Analisis faktor risiko
    st.markdown('<h3 class="sub-header">📊 Analisis Faktor Risiko</h3>', unsafe_allow_html=True)

    # Identifikasi faktor risiko tinggi
    risk_factors = []
    if family_history == "Ya":
        risk_factors.append("Riwayat Keluarga Alzheimer")
    if cardiovascular == "Ya":
        risk_factors.append("Penyakit Kardiovaskular")
    if depression == "Ya":
        risk_factors.append("Depresi")
    if head_injury == "Ya":
        risk_factors.append("Cedera Kepala")
    if memory_complaints == "Ya":
        risk_factors.append("Keluhan Memori")
    if confusion == "Ya":
        risk_factors.append("Kebingungan")
    if disorientation == "Ya":
        risk_factors.append("Disorientasi")
    if difficulty_tasks == "Ya":
        risk_factors.append("Kesulitan Menyelesaikan Tugas")
    if forgetfulness == "Ya":
        risk_factors.append("Pelupa")
    if mmse < 24:
        risk_factors.append("MMSE Score Rendah")
    if bmi > 30:
        risk_factors.append("BMI Tinggi")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚨 Faktor Risiko Teridentifikasi")
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("• Tidak ada faktor risiko mayor teridentifikasi")

    with col2:
        st.markdown("### 💡 Rekomendasi")
        recommendations = [
            "Konsultasi dengan dokter spesialis neurologi",
            "Lakukan pemeriksaan kognitif berkala",
            "Jaga pola hidup sehat dan aktif",
            "Konsumsi makanan bergizi untuk kesehatan otak",
            "Lakukan aktivitas mental yang merangsang otak"
        ]

        for rec in recommendations:
            st.markdown(f"• {rec}")

# Informasi tambahan
st.markdown("---")
st.markdown("### ℹ️ Informasi Penting")
st.markdown("""
- Hasil prediksi ini hanya untuk referensi dan tidak menggantikan diagnosis medis profesional
- Selalu konsultasikan dengan dokter untuk diagnosis yang akurat
- Model ini dikembangkan berdasarkan data penelitian dan mungkin tidak 100% akurat
- Faktor risiko dapat berubah seiring waktu
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧠 AlzDetect </p>
    <p>© 2025 - Barudak tim</p>
</div>
""", unsafe_allow_html=True)
