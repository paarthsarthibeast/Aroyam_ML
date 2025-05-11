import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# --- Path Configuration ---
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.txt"

# --- Page Config ---
st.set_page_config(
    page_title="Heart Failure Readmission Predictor",
    layout="centered",
   page_icon="🫀",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, medical styling
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        color: #0072B5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #444444;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 10px rgba(0,0,0,0.1);
        background-color: white;
        margin-bottom: 20px;
    }
    .section-title {
        color: #0072B5;
        font-size: 1.2rem;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 1px solid #e6e6e6;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #d32f2f;
        padding: 15px;
        border-radius: 4px;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #388e3c;
        padding: 15px;
        border-radius: 4px;
    }
    .stButton>button {
        background-color: #0072B5;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #005a91;
    }
    /* Make radio buttons and selectors more medical-looking */
    div[data-baseweb="select"] > div {
        border-radius: 4px;
    }
    /* Add breathing room to form elements */
    div.row-widget.stRadio > div {
        margin-bottom: 10px;
    }
    /* Clean number input fields */
    input[type="number"] {
        border-radius: 4px !important;
    }
    /* Progress bar colors */
    .stProgress > div > div > div > div {
        background-color: #0072B5;
    }
</style>
""", unsafe_allow_html=True)

# --- Load model and threshold ---
@st.cache_resource
def load_model():
    try:
        # Create directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Check if files exist, otherwise show warning
        if not MODEL_PATH.exists():
            st.warning(f"Model file not found at {MODEL_PATH}. Using mock model for demonstration.")
            # For demo purposes, return a mock model
            class MockModel:
                def __init__(self):
                    self.feature_names_in_ = [
                        'scaled_age', 'scaled_admit_hour', 'scaled_admit_day', 
                        'scaled_avg_lab_value', 'scaled_cpt_code_count', 'scaled_drg_severity',
                        'gender_M', 'gender_F', 'ethnicity_group_Non-Caucasian',
                        'insurance_type_Medicare', 'insurance_type_Medicaid', 
                        'insurance_type_Private', 'insurance_type_Government', 
                        'insurance_type_Self_Pay', 'admission_type_EMERGENCY',
                        'admission_type_URGENT', 'admission_type_ELECTIVE', 
                        'admission_type_NEWBORN', 'icd9_category_Cardiovascular'
                    ]
                def predict_proba(self, X):
                    return np.array([[0.7, 0.3]])
            
            model = MockModel()
            best_threshold = 0.3
        else:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            
            if not THRESHOLD_PATH.exists():
                st.warning(f"Threshold file not found at {THRESHOLD_PATH}. Using default threshold.")
                best_threshold = 0.5
            else:
                with open(THRESHOLD_PATH, "r") as f:
                    best_threshold = float(f.read().strip())
        
        return model, best_threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, 0.5

model, best_threshold = load_model()

# --- Sidebar Info ---
with st.sidebar:
    # Display a simple header instead of an image
    st.markdown("""
    <div style="background-color:#0072B5; padding:10px; border-radius:5px; margin-bottom:10px">
    <h2 style="color:white; text-align:center; font-size:50px; margin:0">❤️</h2>
</div>
    """
    , unsafe_allow_html=True)
    st.title("Heart Failure Readmission Predictor")
    
    st.markdown("### About")
    st.markdown(
        """
        This clinical tool helps predict the risk of **30-day hospital readmission** 
        for heart failure patients using patient data and clinical indicators.
        
        The prediction is based on a machine learning model trained on historical
        patient outcomes.
        """
    )
    
    with st.expander("Model Information"):
        st.markdown("""
        - **Algorithm**: Random Forest classifier
        - **Probability Threshold**: Optimized for clinical relevance
        - **File Paths**:
          - Model: models/random_forest_model.pkl
          - Threshold: models/best_threshold.txt
        """)
    
    st.markdown("### Clinical Disclaimer")
    st.info(
        "This tool is designed to support clinical decision-making only. "
        "All treatment decisions should be made by qualified healthcare "
        "professionals based on comprehensive patient evaluation."
    )
    
    st.markdown("---")
    st.caption("© 2025 Heart Failure Risk Prediction Tool")

# --- App Title ---
st.markdown("<h1 class='main-title'>🫀 Heart Failure Readmission Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict 30-day readmission risk for heart failure patients</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🩺 Risk Assessment", "📚 User Guide"])

# --- Main Content ---
with tab1:
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## 📝Patient Information", unsafe_allow_html=True)

    # Input Form with professional medical styling
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧑‍⚕️ Demographics")
            age = st.slider("Patient Age", 18, 100, 65)
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            ethnicity_group = st.selectbox("Ethnicity", ["Caucasian", "Non-Caucasian"])
        
        with col2:
            st.markdown("#### 🩺 Clinical Information")
            icd9_category_cardiovascular = st.radio(
                "Cardiovascular Condition (ICD-9)", 
                options=[0, 1], 
                format_func=lambda x: "Present" if x == 1 else "Absent",
                horizontal=True
            )
            avg_lab_value = st.number_input("Average Lab Value", min_value=0.0, value=0.0, step=0.1)
            cpt_code_count = st.slider("Number of Procedures (CPT Codes)", 0, 20, 2)
        
        st.markdown("<h4 class='section-title'> 🏥 Admission Details </h4>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            admit_hour = st.slider("Admission Hour", 0, 23, 12, help="Hour of day when patient was admitted (24h format)")
            admit_day = st.selectbox("Day of Admission", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
        with col4:
            insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Government", "Self Pay"])
            admission_type = st.selectbox("Admission Type", ["EMERGENCY", "URGENT", "ELECTIVE", "NEWBORN"])
            drg_severity = st.slider("DRG Severity", 0, 4, 1, help="Diagnosis Related Group severity (0-4)")
        
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            submitted = st.form_submit_button("Generate Risk Assessment")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Feature Encoding and Prediction ---
    if submitted:
        # Create progress indicator
        with st.spinner("Analyzing patient data..."):
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            input_dict = {
                'scaled_age': age,
                'scaled_admit_hour': admit_hour,
                'scaled_admit_day': day_map[admit_day],
                'scaled_avg_lab_value': avg_lab_value,
                'scaled_cpt_code_count': cpt_code_count,
                'scaled_drg_severity': drg_severity,
                'gender_M': 1 if gender == 'Male' else 0,
                'gender_F': 1 if gender == 'Female' else 0,
                'ethnicity_group_Non-Caucasian': 1 if ethnicity_group == 'Non-Caucasian' else 0,
                'insurance_type_Medicare': int(insurance_type == "Medicare"),
                'insurance_type_Medicaid': int(insurance_type == "Medicaid"),
                'insurance_type_Private': int(insurance_type == "Private"),
                'insurance_type_Government': int(insurance_type == "Government"),
                'insurance_type_Self_Pay': int(insurance_type == "Self Pay"),
                'admission_type_EMERGENCY': int(admission_type == "EMERGENCY"),
                'admission_type_URGENT': int(admission_type == "URGENT"),
                'admission_type_ELECTIVE': int(admission_type == "ELECTIVE"),
                'admission_type_NEWBORN': int(admission_type == "NEWBORN"),
                'icd9_category_Cardiovascular': icd9_category_cardiovascular
            }

            input_df = pd.DataFrame([input_dict])

            # Fill in any missing features with 0
            if hasattr(model, 'feature_names_in_'):
                for col in model.feature_names_in_:
                    if col not in input_df.columns:
                        input_df[col] = 0  # Assume 0 for missing binary features
                input_df = input_df[model.feature_names_in_]  # Ensure correct order

            # Prediction
            prob = model.predict_proba(input_df)[:, 1][0]
            prediction = int(prob >= best_threshold)
        
        # Display prediction results
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-title'>Risk Assessment Results</h3>", unsafe_allow_html=True)
        
        # Patient summary
        st.markdown("##### Patient Summary")
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        col_sum1.metric("Age", f"{age} years")
        col_sum2.metric("Gender", gender)
        col_sum3.metric("Admission", admission_type)
        col_sum4.metric("DRG Severity", f"{drg_severity}/4")
        
        # Risk visualization
        st.markdown("##### Readmission Risk Probability")
        
        
        # Clinical assessment and recommendations
        if prediction:
            
            st.subheader("🚨 High Risk of Readmission")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("**Clinical Assessment:**")
                st.markdown("""
                - Patient shows elevated risk for 30-day readmission
                - Close monitoring recommended
                """)
            
            with col_rec2:
                st.markdown("**Recommended Actions:**")
                st.markdown("""
                - Schedule follow-up within 7-10 days
                - Review medication adherence plan
                - Consider home health services
                - Evaluate social support resources
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            
            st.subheader("✅ Low Risk of Readmission")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("**Clinical Assessment:**")
                st.markdown("""
                - Patient shows low risk for 30-day readmission
                - Standard follow-up appropriate
                """)
            
            with col_rec2:
                st.markdown("**Recommended Actions:**")
                st.markdown("""
                - Schedule routine follow-up (2-4 weeks)
                - Standard discharge instructions
                - Provide educational materials
                - Normal monitoring protocol
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional information
        with st.expander("View Clinical Details"):
            st.markdown("##### Risk Factors Analysis")
            
            # Create two columns for risk factors
            factor_col1, factor_col2 = st.columns(2)
            
            with factor_col1:
                st.markdown("**Demographics & Timing**")
                factors_demo = pd.DataFrame({
                    'Factor': ['Age', 'Gender', 'Ethnicity', 'Admission Day', 'Admission Hour'],
                    'Value': [age, gender, ethnicity_group, admit_day, f"{admit_hour}:00"]
                })
                st.dataframe(factors_demo, hide_index=True, use_container_width=True)
            
            with factor_col2:
                st.markdown("**Clinical & Administrative**")
                factors_clinical = pd.DataFrame({
                    'Factor': ['Insurance', 'Admission Type', 'DRG Severity', 'CPT Codes', 'CV ICD9'],
                    'Value': [
                        insurance_type, 
                        admission_type, 
                        drg_severity, 
                        cpt_code_count, 
                        "Present" if icd9_category_cardiovascular == 1 else "Absent"
                    ]
                })
                st.dataframe(factors_clinical, hide_index=True, use_container_width=True)
            
            st.caption(f"Model threshold for high risk classification: {best_threshold:.4f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Print date/time of assessment
        st.caption("Assessment generated on: May 11, 2025")
    else:
        # Show placeholder when no prediction has been made
        st.info("👆 Fill in the patient information above and click 'Generate Risk Assessment' to view results.")

# Footer
st.markdown("---")
st.caption("This tool is designed to assist healthcare providers in identifying patients who may benefit from enhanced care coordination or discharge planning.")
# User Guide Tab
with tab2:
    st.markdown("## 📚 User Guide")
    
    st.markdown("### How to Use This Tool")
    st.markdown("""
    1. **Enter Patient Data**: Fill in all required fields with the patient's information.
    2. **Submit**: Click the 'Predict Risk' button to generate the readmission risk assessment.
    3. **Interpret Results**: Review the risk score and recommendations for patient care planning.
    """)
    
    st.markdown("### Input Fields Explained")
    
    with st.expander("Patient Demographics"):
        st.markdown("""
        - **Age**: Patient's age in years
        - **Gender**: Patient's gender (Male/Female)
        - **Ethnicity**: Patient's ethnic group (Caucasian/Non-Caucasian)
        """)
        
    with st.expander("Clinical Data"):
        st.markdown("""
        - **Average Lab Value**: Mean of normalized lab values
        - **CPT Code Count**: Number of procedures performed
        - **Cardiovascular ICD-9 Code**: Presence of cardiovascular diagnosis code
        - **DRG Severity**: Diagnosis Related Group severity score (0-4)
        """)
        
    with st.expander("Admission Information"):
        st.markdown("""
        - **Admission Hour**: Hour of day when patient was admitted (0-23)
        - **Admission Day**: Day of week when patient was admitted
        - **Admission Type**: Type of hospital admission
        - **Insurance Type**: Patient's insurance coverage
        """)
    
    st.markdown("### Model Information")
    st.markdown("""
    This application uses a Random Forest classifier trained on historical heart failure patient data. 
    The model analyzes multiple patient factors to estimate the probability of readmission within 30 days 
    of discharge.
    
    The decision threshold has been optimized to balance sensitivity and specificity for clinical utility.
    """)
    
    st.warning("""
    **Important:** This tool is designed as a clinical decision support system to assist healthcare 
    providers in identifying patients who may benefit from enhanced care coordination or discharge 
    planning. It should never replace clinical judgment.
    """)