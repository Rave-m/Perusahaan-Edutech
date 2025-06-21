import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Fungsi untuk prediksi input user
def user_input_features():
    st.markdown("### 📋 Data Mahasiswa")
    
    # Tab untuk kategori input yang berbeda
    tab1, tab2 = st.tabs(["📚 Data Akademik", "👤 Data Personal"])
    
    with tab1:
        st.subheader("Nilai dan Mata Kuliah")
        col1, col2 = st.columns(2)
        
        with col1:
            Admission_grade = st.slider('Nilai Masuk', 0.0, 200.0, 120.0, 0.1, 
                                    help="Nilai ujian masuk perguruan tinggi")
            Previous_qualification_grade = st.slider('Nilai Kualifikasi Sebelumnya', 0.0, 200.0, 120.0, 0.1, 
                                    help="Nilai dari pendidikan sebelumnya")
        
        with col2:
            Curricular_units_1st_sem_grade = st.slider('Nilai Rata-rata Semester 1', 0.0, 20.0, 10.0, 0.1)
            Curricular_units_2nd_sem_grade = st.slider('Nilai Rata-rata Semester 2', 0.0, 20.0, 10.0, 0.1)
        
        st.markdown("---")
        
        with st.expander("📊 Detail Jumlah Mata Kuliah"):
            col3, col4 = st.columns(2)
            with col3:
                Curricular_units_1st_sem_enrolled = st.number_input('Mata Kuliah Terdaftar (Sem 1)', min_value=0, max_value=10, value=6)
                Curricular_units_1st_sem_approved = st.number_input('Mata Kuliah Lulus (Sem 1)', min_value=0, max_value=10, value=5)
            with col4:
                Curricular_units_2nd_sem_enrolled = st.number_input('Mata Kuliah Terdaftar (Sem 2)', min_value=0, max_value=10, value=6)
                Curricular_units_2nd_sem_approved = st.number_input('Mata Kuliah Lulus (Sem 2)', min_value=0, max_value=10, value=5)
    
    with tab2:
        col5, col6 = st.columns(2)
        
        with col5:
            Age_at_enrollment = st.slider('Usia saat Masuk', 17, 70, 20)
            Gender = st.radio('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
            Gender = 1 if Gender == 'Laki-laki' else 0
            
        with col6:
            Application_mode = st.selectbox('Jalur Pendaftaran', 
                                        options=[1, 17, 15, 39], 
                                        format_func=lambda x: {1: 'Reguler', 17: 'Khusus', 15: 'Transfer', 39: 'Lainnya'}.get(x, str(x)))
            Displaced = st.radio('Tinggal di Luar Kota Kampus?', ['Ya', 'Tidak'])
            Displaced = 1 if Displaced == 'Ya' else 0
        
        st.markdown("---")
        
        col7, col8 = st.columns(2)
        with col7:
            Debtor = st.radio('Memiliki Tunggakan?', ['Ya', 'Tidak'])
            Debtor = 1 if Debtor == 'Ya' else 0
            
        with col8:
            Tuition_fees_up_to_date = st.radio('SPP Terbayar Tepat Waktu?', ['Ya', 'Tidak'])
            Tuition_fees_up_to_date = 1 if Tuition_fees_up_to_date == 'Ya' else 0
            
        Scholarship_holder = st.radio('Penerima Beasiswa?', ['Ya', 'Tidak'])
        Scholarship_holder = 1 if Scholarship_holder == 'Ya' else 0
    
    # Hitung variabel turunan secara otomatis
    Total_enrolled_units = Curricular_units_1st_sem_enrolled + Curricular_units_2nd_sem_enrolled
    Total_approved_unit = Curricular_units_1st_sem_approved + Curricular_units_2nd_sem_approved
    Approval_rate = (Total_approved_unit / Total_enrolled_units * 100) if Total_enrolled_units > 0 else 0
    Average_grade = (Curricular_units_1st_sem_grade + Curricular_units_2nd_sem_grade) / 2

    # Create a DataFrame with feature names to avoid RandomForest feature names error
    feature_names = [
        'Application_mode', 'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 
        'Age_at_enrollment', 'Previous_qualification_grade', 'Admission_grade', 'Displaced',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 
        'Total_enrolled_units', 'Total_approved_unit', 'Approval_rate', 'Average_grade'
    ]
    
    input_values = [[
        Application_mode, Debtor, Tuition_fees_up_to_date, Gender, Scholarship_holder,
        Age_at_enrollment, Previous_qualification_grade, Admission_grade, Displaced,
        Curricular_units_1st_sem_enrolled, Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade,
        Curricular_units_2nd_sem_enrolled, Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade,
        Total_enrolled_units, Total_approved_unit, Approval_rate, Average_grade
    ]]
    
    features = pd.DataFrame(input_values, columns=feature_names)
    
    return features

# Fungsi untuk prediksi menggunakan model RF
def predict_student_status(input_features, model, scaller):
    """
    Predict student status using the trained Random Forest model
    
    Args:
        input_features: DataFrame with the correct feature names and format
        model: The loaded machine learning model
        scaller: The loaded scaler for feature normalization
    
    Returns:
        prediction: predicted class (0: Dropout, 1: Enrolled, 2: Graduate)
        probabilities: probability for each class
    """
    # Get feature names before scaling
    feature_names = input_features.columns
    
    # Scale the input features
    scaled_features = scaller.transform(input_features)
    
    # Convert back to DataFrame with the same feature names to avoid warning
    scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
    
    # Make prediction
    prediction = model.predict(scaled_df)
    
    # Get probabilities if available
    try:
        probabilities = model.predict_proba(scaled_df)[0]
    except:
        probabilities = None
        
    return prediction[0], probabilities