import pandas as pd
import plotly.express as px

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
