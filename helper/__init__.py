from .dashboard import (create_status_distribution, create_course_success_rate, 
                        create_age_distribution, create_grade_analysis,
                        create_economic_impact, create_scholarship_impact)

from .prediction import user_input_features, predict_student_status

# Expose all functions
__all__ = [
    # Dashboard functions
    'create_status_distribution', 'create_course_success_rate', 
    'create_age_distribution', 'create_grade_analysis',
    'create_economic_impact', 'create_scholarship_impact',
    
    # Prediction functions
    'user_input_features', 'predict_student_status'
]
