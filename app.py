import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Import modular components
from helper import (
    # Dashboard functions
    create_status_distribution, create_course_success_rate, 
    create_age_distribution, create_grade_analysis,
    create_economic_impact, create_scholarship_impact,
    
    # Prediction functions
    user_input_features, predict_student_status
)

# Load model, scaller, and data
try:
    model_path = 'model/rf_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model_loaded = True
    else:
        st.error(f"Model file not found: {model_path}")
        model_loaded = False
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False
    
try:
    scaller_path = 'model/scaller.pkl'
    if os.path.exists(scaller_path):
        scaller = joblib.load(scaller_path)
        scaller_loaded = True
    else:
        st.error(f"Scaller file not found: {scaller_path}")
        scaller_loaded = False
except Exception as e:
    st.error(f"Error loading scaller: {str(e)}")
    scaller_loaded = False

# Load dataset for visualization purposes only
data = pd.read_csv("data/data_students.csv", delimiter=",")

# Map course codes to names
course_mapping = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
}
data['Course_Name'] = data['Course'].map(course_mapping)

# UI Setup
st.set_page_config(page_title="Mahasiswa Analytics", layout="wide")

# Sidebar for navigation
st.sidebar.title("📋 Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Dashboard", "🔮 Prediksi"])
st.sidebar.markdown("---")

# Author info in sidebar
st.sidebar.markdown("### 👨‍💻 Developed by")
st.sidebar.info("Wahid Hasim")
st.sidebar.markdown("---")

# Main Application Logic
if page == "📊 Dashboard":
    st.title("📊 Dashboard Analisis Mahasiswa")
    st.write("Visualisasi data performa dan status mahasiswa")
    
    # Dashboard filters
    st.sidebar.header("🔍 Filter Dashboard")
    
    # Course filter
    all_courses = ['Semua Program'] + sorted(data['Course_Name'].unique().tolist())
    selected_course = st.sidebar.selectbox('Program Studi:', all_courses)
    
    # Gender filter
    gender_options = ['Semua', 'Laki-laki', 'Perempuan']
    selected_gender = st.sidebar.selectbox('Jenis Kelamin:', gender_options)
    
    # Age range filter
    age_min = int(data['Age_at_enrollment'].min())
    age_max = int(data['Age_at_enrollment'].max())
    age_range = st.sidebar.slider(
        'Rentang Usia:',
        min_value=age_min, 
        max_value=age_max, 
        value=(age_min, age_max)
    )
    
    # Apply filters
    filtered_data = data.copy()
    
    if selected_course != 'Semua Program':
        filtered_data = filtered_data[filtered_data['Course_Name'] == selected_course]
    
    if selected_gender != 'Semua':
        gender_map = {'Laki-laki': 1, 'Perempuan': 0}
        filtered_data = filtered_data[filtered_data['Gender'] == gender_map[selected_gender]]
    
    filtered_data = filtered_data[(filtered_data['Age_at_enrollment'] >= age_range[0]) & 
                                (filtered_data['Age_at_enrollment'] <= age_range[1])]
    
    # ui Show filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data yang ditampilkan:** {len(filtered_data)} mahasiswa")
    
    # Key metrics
    st.markdown("### 📌 Metrik Utama")
    
    total_students = len(filtered_data)
    dropout_count = len(filtered_data[filtered_data['Status'] == 'Dropout'])
    graduate_count = len(filtered_data[filtered_data['Status'] == 'Graduate'])
    enrolled_count = len(filtered_data[filtered_data['Status'] == 'Enrolled'])
    
    dropout_rate = (dropout_count / total_students * 100) if total_students > 0 else 0
    graduate_rate = (graduate_count / total_students * 100) if total_students > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Mahasiswa", f"{total_students}")
    with col2:
        st.metric("Tingkat Dropout", f"{dropout_rate:.1f}%")
    with col3:
        st.metric("Tingkat Kelulusan", f"{graduate_rate:.1f}%")
    with col4:
        st.metric("Masih Terdaftar", f"{enrolled_count}")
    
    # Charts
    st.markdown("---")
    
    # Row 1: Status distribution & Grade analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_status_distribution(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_grade_analysis(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Age distribution & Scholarship impact
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_age_distribution(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = create_scholarship_impact(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Economic impact & Course success rate
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_economic_impact(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = create_course_success_rate(filtered_data)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prediksi":
    st.title("🔮 Prediksi Status Mahasiswa")
    
    # Periksa apakah model berhasil dimuat
    if not model_loaded:
        st.error("⚠️ Model prediksi tidak dapat dimuat. Fitur prediksi tidak tersedia.")
        st.info("Pastikan file model/rf_model.pkl tersedia dan valid.")
    else:
        st.write("Masukkan data untuk memprediksi kemungkinan status mahasiswa")
        
        # Ambil input
        input_data = user_input_features()
          # Prediksi
        st.markdown("### 🔮 Hasil Prediksi")
        if st.button('Prediksi Status Mahasiswa', use_container_width=True):
            # Gunakan fungsi prediksi untuk memastikan penggunaan model langsung
            prediction, probabilities = predict_student_status(input_data, model, scaller)
            status_map = {0: 'Dropout', 1: 'Masih Terdaftar', 2: 'Lulus'}
            prediction_label = status_map.get(prediction, "Tidak Diketahui")
            
            # Tampilkan hasil dengan warna berbeda berdasarkan prediksi
            if prediction == 0:  # Dropout
                st.error(f"### 🚫 Prediksi: DROPOUT")
                st.info("Mahasiswa ini diprediksi akan dropout. Disarankan untuk memberikan perhatian khusus dan bimbingan.")
            elif prediction == 1:  # Enrolled
                st.info(f"### ⏳ Prediksi: MASIH TERDAFTAR")
                st.success("Mahasiswa ini diprediksi akan tetap terdaftar. Terus pantau perkembangannya.")
            else:  # Graduated
                st.success(f"### 🎓 Prediksi: LULUS")
                st.balloons()
                st.info("Mahasiswa ini diprediksi akan lulus dengan baik. Pertahankan performa akademiknya.")
            
            # Tampilkan probabilitas jika tersedia
            if probabilities is not None:
                st.markdown("#### Tingkat Kepercayaan Prediksi:")
                
                col_prob1, col_prob2, col_prob3 = st.columns(3)
                with col_prob1:
                    st.progress(probabilities[0])
                    st.caption(f"Dropout: {probabilities[0]:.1%}")
                with col_prob2:
                    st.progress(probabilities[1])
                    st.caption(f"Masih Terdaftar: {probabilities[1]:.1%}")
                with col_prob3:
                    st.progress(probabilities[2])
                    st.caption(f"Lulus: {probabilities[2]:.1%}")
                
                # Tampilkan interpretasi hasil prediksi
                st.markdown("### 📊 Interpretasi Hasil")
                
                # Mencari kelas dengan probabilitas tertinggi
                max_prob_idx = np.argmax(probabilities)
                max_prob_value = probabilities[max_prob_idx]
                
                if max_prob_value > 0.7:
                    st.success(f"Prediksi memiliki tingkat kepercayaan tinggi ({max_prob_value:.1%})")
                elif max_prob_value > 0.5:
                    st.info(f"Prediksi memiliki tingkat kepercayaan sedang ({max_prob_value:.1%})")
                else:
                    st.warning(f"Prediksi memiliki tingkat kepercayaan rendah ({max_prob_value:.1%})")
                    st.write("Disarankan untuk memperhatikan faktor-faktor lain dalam pengambilan keputusan.")