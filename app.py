import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load model and data
model = joblib.load('model/rf_model.pkl')
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

# Functions for dashboard visualizations
def create_status_distribution(df):
    # Status distribution
    status_counts = df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    # Translate status
    status_mapping = {'Dropout': 'Dropout', 'Graduate': 'Lulus', 'Enrolled': 'Terdaftar'}
    status_counts['Status'] = status_counts['Status'].map(status_mapping)
    
    fig = px.pie(
        status_counts, 
        values='Count', 
        names='Status', 
        title='Distribusi Status Mahasiswa',
        color='Status',
        color_discrete_map={
            'Dropout': '#FF6B6B', 
            'Terdaftar': '#4ECDC4', 
            'Lulus': '#59CD90'
        },
        hole=0.4
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    return fig

def create_course_success_rate(df):
    # Group by course and calculate percentage of graduates, enrollees and dropouts
    course_stats = df.groupby(['Course_Name', 'Status']).size().unstack().fillna(0)
    
    if 'Graduate' not in course_stats.columns:
        course_stats['Graduate'] = 0
    if 'Dropout' not in course_stats.columns:
        course_stats['Dropout'] = 0
    if 'Enrolled' not in course_stats.columns:
        course_stats['Enrolled'] = 0
    
    # Calculate total and success rate
    course_stats['Total'] = course_stats.sum(axis=1)
    course_stats['Success_Rate'] = ((course_stats['Graduate'] + course_stats['Enrolled']) / course_stats['Total'] * 100)
    
    # Sort by success rate
    course_stats = course_stats.sort_values('Success_Rate', ascending=True).reset_index().tail(10)
    
    fig = px.bar(
        course_stats,
        x='Success_Rate',
        y='Course_Name',
        orientation='h',
        title='Top 10 Program Studi dengan Tingkat Kesuksesan Tertinggi',
        labels={'Success_Rate': 'Tingkat Kesuksesan (%)', 'Course_Name': 'Program Studi'},
        color_discrete_sequence=['#4ECDC4']
    )
    return fig

def create_age_distribution(df):
    # Age distribution by status
    age_status = df[['Age_at_enrollment', 'Status']].copy()
    
    # Translate status
    status_mapping = {'Dropout': 'Dropout', 'Graduate': 'Lulus', 'Enrolled': 'Terdaftar'}
    age_status['Status'] = age_status['Status'].map(status_mapping)
    
    fig = px.histogram(
        age_status, 
        x='Age_at_enrollment', 
        color='Status',
        nbins=20,
        title='Distribusi Usia berdasarkan Status',
        labels={'Age_at_enrollment': 'Usia saat Pendaftaran', 'count': 'Jumlah Mahasiswa'},
        color_discrete_map={
            'Dropout': '#FF6B6B', 
            'Terdaftar': '#4ECDC4', 
            'Lulus': '#59CD90'
        }
    )
    return fig

def create_grade_analysis(df):
    # Create grade visualization
    grade_data = df[['Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade', 'Status']].copy()
    grade_data['Average_Grade'] = (grade_data['Curricular_units_1st_sem_grade'] + grade_data['Curricular_units_2nd_sem_grade']) / 2
    
    # Translate status
    status_mapping = {'Dropout': 'Dropout', 'Graduate': 'Lulus', 'Enrolled': 'Terdaftar'}
    grade_data['Status'] = grade_data['Status'].map(status_mapping)
    
    fig = px.box(
        grade_data,
        x='Status',
        y='Average_Grade',
        color='Status',
        title='Distribusi Nilai Rata-rata berdasarkan Status',
        labels={'Average_Grade': 'Nilai Rata-rata', 'Status': 'Status Mahasiswa'},
        color_discrete_map={
            'Dropout': '#FF6B6B', 
            'Terdaftar': '#4ECDC4', 
            'Lulus': '#59CD90'
        }
    )
    return fig

def create_economic_impact(df):
    # Economic pressure score vs status
    eco_data = df[['Economic_pressure_score', 'Status', 'Debtor']].copy()
    
    # Translate status
    status_mapping = {'Dropout': 'Dropout', 'Graduate': 'Lulus', 'Enrolled': 'Terdaftar'}
    eco_data['Status'] = eco_data['Status'].map(status_mapping)
    eco_data['Has_Debt'] = eco_data['Debtor'].map({0: 'Tidak Memiliki Hutang', 1: 'Memiliki Hutang'})
    
    fig = px.scatter(
        eco_data, 
        x='Economic_pressure_score', 
        y='Status', 
        color='Has_Debt',
        title='Pengaruh Tekanan Ekonomi terhadap Status',
        labels={'Economic_pressure_score': 'Skor Tekanan Ekonomi', 'Status': 'Status Mahasiswa'},
        color_discrete_sequence=['#4ECDC4', '#FF6B6B']
    )
    return fig

def create_scholarship_impact(df):
    # Scholarship impact
    scholar_data = df.groupby(['Scholarship_holder', 'Status']).size().unstack().fillna(0)
    
    if 'Graduate' not in scholar_data.columns:
        scholar_data['Graduate'] = 0
    if 'Dropout' not in scholar_data.columns:
        scholar_data['Dropout'] = 0
    if 'Enrolled' not in scholar_data.columns:
        scholar_data['Enrolled'] = 0
        
    # Calculate percentages
    scholar_data['Total'] = scholar_data.sum(axis=1)
    for col in ['Graduate', 'Dropout', 'Enrolled']:
        scholar_data[f'{col}_Pct'] = scholar_data[col] / scholar_data['Total'] * 100
    
    # Create data for visualization
    scholarship_status = []
    labels = {0: 'Tanpa Beasiswa', 1: 'Dengan Beasiswa'}
    statuses = {'Graduate': 'Lulus', 'Dropout': 'Dropout', 'Enrolled': 'Terdaftar'}
    
    for scholarship in [0, 1]:
        for status in ['Graduate', 'Dropout', 'Enrolled']:
            scholarship_status.append({
                'Scholarship': labels[scholarship],
                'Status': statuses[status],
                'Percentage': scholar_data.loc[scholarship, f'{status}_Pct']
            })
    
    scholarship_df = pd.DataFrame(scholarship_status)
    
    fig = px.bar(
        scholarship_df, 
        x='Scholarship', 
        y='Percentage', 
        color='Status',
        title='Dampak Beasiswa terhadap Status Mahasiswa',
        labels={'Percentage': 'Persentase (%)', 'Scholarship': 'Status Beasiswa'},
        color_discrete_map={
            'Dropout': '#FF6B6B', 
            'Terdaftar': '#4ECDC4', 
            'Lulus': '#59CD90'
        },
        barmode='group'
    )
    return fig

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

    # Tampilkan metrik-metrik perhitungan
    st.markdown("### 📈 Metrik Performa")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    with col_metrics1:
        st.metric("Total MK Terdaftar", f"{Total_enrolled_units}")
    with col_metrics2:
        st.metric("Total MK Lulus", f"{Total_approved_unit}")
    with col_metrics3:
        st.metric("Tingkat Kelulusan", f"{Approval_rate:.1f}%")
    
    # Tampilkan gauge untuk rata-rata nilai
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = Average_grade,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Rata-rata Nilai"},
        gauge = {
            'axis': {'range': [0, 20]},
            'bar': {'color': "#5D9A96" if Average_grade >= 10 else "#D2546C"},
            'steps': [
                {'range': [0, 10], 'color': "#FFE2E2"},
                {'range': [10, 15], 'color': "#C9FFE2"},
                {'range': [15, 20], 'color': "#BAFFC9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 10
            }
        }
    ))

    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

    # Create a DataFrame with feature names to avoid RandomForest feature names error
    feature_names = [
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Admission_grade', 'Previous_qualification_grade', 'Age_at_enrollment',
        'Tuition_fees_up_to_date', 'Scholarship_holder', 'Gender', 'Debtor',
        'Application_mode', 'Displaced', 'Total_enrolled_units', 'Total_approved_unit', 'Approval_rate', 'Average_grade'
    ]
    
    input_values = [[
        Curricular_units_2nd_sem_enrolled, Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade,
        Curricular_units_1st_sem_enrolled, Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade,
        Admission_grade, Previous_qualification_grade, Age_at_enrollment,
        Tuition_fees_up_to_date, Scholarship_holder, Gender, Debtor,
        Application_mode, Displaced, Total_enrolled_units, Total_approved_unit, Approval_rate, Average_grade
    ]]
    
    features = pd.DataFrame(input_values, columns=feature_names)
    
    return features

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
    
    # Show filter summary
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
    st.write("Masukkan data untuk memprediksi kemungkinan status mahasiswa")
    
    # Ambil input
    input_data = user_input_features()
    
    # Prediksi
    st.markdown("### 🔮 Hasil Prediksi")
    if st.button('Prediksi Status Mahasiswa', use_container_width=True):
        prediction = model.predict(input_data)
        status_map = {0: 'Dropout', 1: 'Masih Terdaftar', 2: 'Lulus'}
        prediction_label = status_map.get(prediction[0], "Tidak Diketahui")
        
        # Tampilkan hasil dengan warna berbeda berdasarkan prediksi
        if prediction[0] == 0:  # Dropout
            st.error(f"### 🚫 Prediksi: DROPOUT")
            st.info("Mahasiswa ini diprediksi akan dropout. Disarankan untuk memberikan perhatian khusus dan bimbingan.")
        elif prediction[0] == 1:  # Enrolled
            st.info(f"### ⏳ Prediksi: MASIH TERDAFTAR")
            st.success("Mahasiswa ini diprediksi akan tetap terdaftar. Terus pantau perkembangannya.")
        else:  # Graduated
            st.success(f"### 🎓 Prediksi: LULUS")
            st.balloons()
            st.info("Mahasiswa ini diprediksi akan lulus dengan baik. Pertahankan performa akademiknya.")
        
        # Tampilkan probabilitas (opsional, jika model mendukung)
        try:
            proba = model.predict_proba(input_data)[0]
            st.markdown("#### Tingkat Kepercayaan Prediksi:")
            
            col_prob1, col_prob2, col_prob3 = st.columns(3)
            with col_prob1:
                st.progress(proba[0])
                st.caption(f"Dropout: {proba[0]:.1%}")
            with col_prob2:
                st.progress(proba[1])
                st.caption(f"Masih Terdaftar: {proba[1]:.1%}")
            with col_prob3:
                st.progress(proba[2])
                st.caption(f"Lulus: {proba[2]:.1%}")
        except:
            pass
