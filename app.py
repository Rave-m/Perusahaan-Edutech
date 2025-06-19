import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load model
model = joblib.load('model/rf_model.pkl')

# Page Configuration
st.set_page_config(page_title="Student Status Prediction", layout="wide")

# Load data
data = pd.read_csv("data/data_students.csv", delimiter=",")

# Add Status columns for easier filtering
data['Status_0'] = (data['Status'] == 0).astype(int)  # Dropout
data['Status_1'] = (data['Status'] == 1).astype(int)  # Enrolled
data['Status_2'] = (data['Status'] == 2).astype(int)  # Graduated
data['Status_New'] = (data['Status'] > 0).astype(int)  # Not Dropout (Enrolled or Graduated)

# Course mapping
category_mapping = {
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
data['Course_Label'] = data['Course'].replace(category_mapping)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page:", ["Dashboard", "Prediction"])

# Helper functions
def add_rating(content):
    return f"""
        <div style='
            height: auto;
            border: 2px solid #ccc;
            border-radius: 5px;
            font-size: 25px;
            padding-bottom: 38px;
            padding-top: 38px;
            background-color: #fffff;
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            '>{content}</div>
        """

def add_card(content):
    return f"""
        <div style='
            height: auto;
            font-size: auto;
            border: 2px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #fffff;
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            line-height: 70px;
            '>{content}</div>
        """

def create_pie_chart(column, title):
    try:
        value_counts = filtered_data[column].value_counts()
        if len(value_counts) > 1:
            names = [False, True]
        else:
            if value_counts.index[0] == 1:
                names = [True]
            else:
                names = [False]
        colors = ['white', '#393939']
        fig = px.pie(
            values=value_counts,
            names=names,
            title=title,
            color_discrete_sequence=colors)
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=10, t=70, b=10),
            title=dict(
                x=0,
                font=dict(size=15),
            ),
        )
        st.plotly_chart(fig)
    except (UnboundLocalError, IndexError) as e:
        st.write("No data available to display.")

# Dashboard Page
if page == "Dashboard":
    st.title('Student Performance Dashboard')
    st.markdown("---")
    
    # Filters section
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_options = ['All', 'Dropout', 'Enrolled', 'Graduated']
        selected_status = st.selectbox('Status', status_options)
    
    with col2:
        course_list = sorted(list(data['Course_Label'].unique()))
        course_list.insert(0, "All Courses")
        selected_course = st.selectbox('Course', course_list)
    
    with col3:
        time_options = ['All', 'Daytime', 'Evening']
        selected_time = st.selectbox('Attendance Time', time_options)
    
    with col4:
        gender_options = ['All', 'Male', 'Female']
        selected_gender = st.selectbox('Gender', gender_options)
    
    # Apply filters
    filtered_data = data.copy()
    
    if selected_status != 'All':
        if selected_status == 'Dropout':
            filtered_data = filtered_data[filtered_data['Status'] == 0]
        elif selected_status == 'Enrolled':
            filtered_data = filtered_data[filtered_data['Status'] == 1]
        elif selected_status == 'Graduated':
            filtered_data = filtered_data[filtered_data['Status'] == 2]
    
    if selected_course != 'All Courses':
        filtered_data = filtered_data[filtered_data['Course_Label'] == selected_course]
    
    if selected_time != 'All':
        if selected_time == 'Daytime':
            filtered_data = filtered_data[filtered_data['Daytime_evening_attendance'] == 1]
        else:
            filtered_data = filtered_data[filtered_data['Daytime_evening_attendance'] == 0]
    
    if selected_gender != 'All':
        if selected_gender == 'Male':
            filtered_data = filtered_data[filtered_data['Gender'] == 1]
        else:
            filtered_data = filtered_data[filtered_data['Gender'] == 0]
    
    st.markdown("---")
    
    # Overview Section
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(filtered_data)
    dropout_count = sum(filtered_data['Status'] == 0)
    enrolled_count = sum(filtered_data['Status'] == 1)
    graduated_count = sum(filtered_data['Status'] == 2)
    
    dropout_rate = dropout_count / total_students * 100 if total_students > 0 else 0
    
    with col1:
        st.metric("Total Students", f"{total_students}")
    with col2:
        st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
    with col3:
        st.metric("Enrolled Students", f"{enrolled_count}")
    with col4:
        st.metric("Graduated Students", f"{graduated_count}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Status Distribution")
        status_counts = filtered_data['Status'].value_counts().sort_index()
        status_labels = ['Dropout', 'Enrolled', 'Graduated']
        
        if not status_counts.empty:
            fig = px.pie(
                values=status_counts, 
                names=[status_labels[i] for i in status_counts.index],
                title="Student Status Distribution",
                color_discrete_sequence=['#FF5252', '#4CAF50', '#2196F3']
            )
            st.plotly_chart(fig)
        else:
            st.write("No data available for selected filters.")
    
    with col2:
        st.subheader("Average Grade by Status")
        try:
            avg_grades = filtered_data.groupby('Status')[['Curricular_units_1st_sem_grade', 
                                                        'Curricular_units_2nd_sem_grade']].mean().reset_index()
            
            if not avg_grades.empty:
                avg_grades['Status'] = avg_grades['Status'].replace({0: 'Dropout', 1: 'Enrolled', 2: 'Graduated'})
                
                fig = px.bar(
                    avg_grades,
                    x='Status',
                    y=['Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade'],
                    barmode='group',
                    labels={
                        'value': 'Average Grade',
                        'variable': 'Semester'
                    },
                    title="Average Grades by Student Status",
                    color_discrete_sequence=['#FFA500', '#9C27B0']
                )
                
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
                
                st.plotly_chart(fig)
            else:
                st.write("No data available for selected filters.")
        except Exception as e:
            st.write("Error creating visualization:", e)
    
    # More visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        try:
            fig = px.histogram(
                filtered_data,
                x='Age_at_enrollment',
                nbins=20,
                title='Age at Enrollment Distribution',
                color_discrete_sequence=['#3F51B5']
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.write("Error creating visualization:", e)
    
    with col2:
        st.subheader("Scholarship by Status")
        try:
            scholarship_by_status = filtered_data.groupby('Status')['Scholarship_holder'].mean().reset_index()
            
            if not scholarship_by_status.empty:
                scholarship_by_status['Status'] = scholarship_by_status['Status'].replace(
                    {0: 'Dropout', 1: 'Enrolled', 2: 'Graduated'}
                )
                scholarship_by_status['Scholarship_holder'] = scholarship_by_status['Scholarship_holder'] * 100
                
                fig = px.bar(
                    scholarship_by_status,
                    x='Status',
                    y='Scholarship_holder',
                    title="Scholarship Holders by Status (%)",
                    text_auto='.1f',
                    color='Status',
                    color_discrete_sequence=['#FF5252', '#4CAF50', '#2196F3']
                )
                
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                fig.update_layout(showlegend=False)
                
                st.plotly_chart(fig)
            else:
                st.write("No data available for selected filters.")
        except Exception as e:
            st.write("Error creating visualization:", e)

# Prediction Page
elif page == "Prediction":
    st.title('Student Status Prediction')
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    This tool predicts whether a student is likely to:
    - **Dropout** (leave without completing the course)
    - **Enrolled** (continue their studies)
    - **Graduated** (successfully complete the course)
    
    Fill in the form below with student data to get a prediction.
    """)
    
    # Input form in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        application_mode = st.number_input('Application Mode', min_value=0)
        gender = st.selectbox('Gender', [('Male', 1), ('Female', 0)], format_func=lambda x: x[0])
        age = st.number_input('Age at Enrollment', min_value=16, max_value=70, value=18)
        displaced = st.selectbox('Displaced (student living away from home)', [('Yes', 1), ('No', 0)], format_func=lambda x: x[0])
        scholarship = st.selectbox('Scholarship Holder', [('Yes', 1), ('No', 0)], format_func=lambda x: x[0])
        debtor = st.selectbox('Debtor', [('Yes', 1), ('No', 0)], format_func=lambda x: x[0])
        tuition_up_to_date = st.selectbox('Tuition Fees Up to Date', [('Yes', 1), ('No', 0)], format_func=lambda x: x[0])
    
    with col2:
        st.subheader("Academic Information")
        prev_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0.0, max_value=200.0, value=120.0)
        admission_grade = st.number_input('Admission Grade', min_value=0.0, max_value=200.0, value=120.0)
        
        st.markdown("#### First Semester")
        units_1st_enrolled = st.number_input('1st Semester - Enrolled Units', min_value=0, max_value=20, value=6)
        units_1st_approved = st.number_input('1st Semester - Approved Units', min_value=0, max_value=20, value=5)
        grade_1st = st.number_input('1st Semester - Grade', min_value=0.0, max_value=20.0, value=12.0)
        
        st.markdown("#### Second Semester")
        units_2nd_enrolled = st.number_input('2nd Semester - Enrolled Units', min_value=0, max_value=20, value=6)
        units_2nd_approved = st.number_input('2nd Semester - Approved Units', min_value=0, max_value=20, value=5)
        grade_2nd = st.number_input('2nd Semester - Grade', min_value=0.0, max_value=20.0, value=12.0)
    
    # Calculate derived features
    total_enrolled = units_1st_enrolled + units_2nd_enrolled
    total_approved = units_1st_approved + units_2nd_approved
    approval_rate = (total_approved / total_enrolled * 100) if total_enrolled > 0 else 0
    average_grade = (grade_1st + grade_2nd) / 2 if (grade_1st + grade_2nd > 0) else 0
    
    # Create input array for prediction
    input_data = np.array([[
        units_2nd_enrolled, units_2nd_approved, grade_2nd,
        units_1st_enrolled, units_1st_approved, grade_1st,
        admission_grade, prev_qualification_grade, age,
        tuition_up_to_date[1], scholarship[1], gender[1], debtor[1],
        application_mode, displaced[1], total_enrolled, total_approved, approval_rate, average_grade
    ]])
    
    # Show prediction
    if st.button('Predict Student Status', use_container_width=True):
        with st.spinner('Predicting...'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            
            status_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduated'}
            prediction_label = status_map.get(prediction[0], "Unknown")
            
            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Prediction box
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: #0066cc;'>Predicted Status: {prediction_label}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Show probabilities
            st.markdown("#### Prediction Confidence")
            proba_df = pd.DataFrame({
                'Status': ['Dropout', 'Enrolled', 'Graduated'],
                'Probability': probability[0] * 100
            })
            
            fig = px.bar(
                proba_df, 
                x='Status', 
                y='Probability',
                text_auto='.1f',
                color='Status',
                color_discrete_sequence=['#FF5252', '#4CAF50', '#2196F3']
            )
            
            fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig.update_layout(yaxis_range=[0, 100])
            
            st.plotly_chart(fig)
            
            # Summary of input
            with st.expander("View Input Summary"):
                input_summary = {
                    'Basic Information': {
                        'Application Mode': application_mode,
                        'Gender': gender[0],
                        'Age': age,
                        'Displaced': displaced[0],
                        'Scholarship Holder': scholarship[0],
                        'Debtor': debtor[0],
                        'Tuition Up to Date': tuition_up_to_date[0]
                    },
                    'Academic Information': {
                        'Previous Qualification Grade': prev_qualification_grade,
                        'Admission Grade': admission_grade,
                        '1st Semester Enrolled Units': units_1st_enrolled,
                        '1st Semester Approved Units': units_1st_approved,
                        '1st Semester Grade': grade_1st,
                        '2nd Semester Enrolled Units': units_2nd_enrolled,
                        '2nd Semester Approved Units': units_2nd_approved,
                        '2nd Semester Grade': grade_2nd
                    },
                    'Calculated Metrics': {
                        'Total Enrolled Units': total_enrolled,
                        'Total Approved Units': total_approved,
                        'Approval Rate': f"{approval_rate:.2f}%",
                        'Average Grade': average_grade
                    }
                }
                
                for category, items in input_summary.items():
                    st.write(f"**{category}**")
                    for key, value in items.items():
                        st.write(f"- {key}: {value}")
    else:
        st.info("Fill in the form and click 'Predict Student Status' to get a prediction.")
