import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import time

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

# Detect system theme and define dynamic colors

# --- Dynamic Theme Detection ---
theme = st.get_option("theme.base")

if theme == "dark":
    TEXT_COLOR = "#FAFAFA"
    METRIC_BG = "#2F2F2F"
    METRIC_LABEL_COLOR = "#000000"  # Black label text in dark mode
    METRIC_VALUE_COLOR = "#FAFAFA"
else:
    TEXT_COLOR = "#000000"
    METRIC_BG = "#f0fdf4"
    METRIC_LABEL_COLOR = "#000000"  # Black label text in light mode
    METRIC_VALUE_COLOR = "#000000"




theme = st.get_option("theme.base")

if theme == "dark":
    BG_COLOR = "#1E1E1E"
    TEXT_COLOR = "#FAFAFA"
    CARD_BG = "#2A2A2A"
    LOW_RISK_COLOR = "#27ae60"
    HIGH_RISK_COLOR = "#e74c3c"
else:
    BG_COLOR = "#FFFFFF"
    TEXT_COLOR = "#000000"
    CARD_BG = "#f0fff4"
    LOW_RISK_COLOR = "#2ecc71"
    HIGH_RISK_COLOR = "#e74c3c"



st.markdown("""
<style>
    /* Base styling */
    * {
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease-in-out;
    }
    
    /* Main heading animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    /* Card entrance animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Main title and subtitle */
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #0072B5;
        text-align: center;
        margin-bottom: 0.3rem;
        animation: pulse 2s infinite;
    }
    
    .heart-icon {
        color: #e74c3c;
        animation: pulse 2s infinite;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    
    /* Card styling */
    .card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: white;
        margin-bottom: 24px;
        border-left: 4px solid #0072B5;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Section titles */
    .section-title {
        color: #0072B5;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e6e6e6;
    }
    
    /* Risk indicator styling */
    .high-risk {
        background-color: #fff5f5;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 8px;
        animation: fadeIn 0.8s ease-out;
        margin-bottom: 50px;
    }
    
    .low-risk {
        background-color: #f0fff4;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 8px;
        animation: fadeIn 0.8s ease-out;
        margin-bottom: 50px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #0072B5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 2px 5px rgba(0,114,181,0.3);
        
    }
    
    .stButton>button:hover {
        background-color: #005a91;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,114,181,0.4);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Form input styling */
    div[data-baseweb="select"] div {
        border-radius: 8px;
        border-color: #e6e6e6;
    }
    
    div[data-baseweb="select"] div:hover {
        border-color: #0072B5;
    }
    
    .stSlider > div {
        padding-top: 0.5rem;
        padding-bottom: 1.5rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0072B5 0%, #00add8 100%);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    ddiv[data-testid="stMetric"] > label{
        
        font-size: 2rem;
        font-weight: bold;
            
    }
   
   
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #0072B5;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Risk score visualization */
    .risk-container {
        position: relative;
        width: 100%;
        height: 30px;
        background-color: #f0f0f0;
        border-radius: 15px;
        overflow: hidden;
        margin: 20px 0;
    }
    
    .risk-bar {
        position: absolute;
        height: 100%;
        left: 0;
        border-radius: 15px;
        transition: width 1s ease-in-out;
    }
    
    .risk-label {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: bold;
        text-shadow: 0 0 3px rgba(0,0,0,0.5);
    }
    
    /* Make radio buttons more modern */
    div.st-bf {
        border-radius: 8px;
    }
    
    div.st-cc {
        gap: 0.5rem;
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


with st.sidebar:
    # header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0072B5 0%, #00add8 100%); 
         padding:15px; border-radius:10px; margin-bottom:15px; 
         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
         animation: fadeIn 0.8s ease-out;">
        <h2 style="color:white; text-align:center; margin:0; display:flex; align-items:center; justify-content:center;">
            <span class="heart-icon" style="font-size:1.5em; margin-right:10px;">❤️</span>
            Heart Failure Predictor
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="animation: fadeIn 1s ease-out;">
        <h3 style="color:#0072B5; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">About</h3>
    </div>
    """, unsafe_allow_html=True)
    
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
        - **Model Path**: models/random_forest_model.pkl
        """)
    
    st.markdown("""
    <div style="animation: fadeIn 1.2s ease-out;">
        <h3 style="color:#0072B5; border-bottom:2px solid #e6e6e6; padding-bottom:8px;">Clinical Disclaimer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(
        "This tool is designed to support clinical decision-making only. "
        "All treatment decisions should be made by qualified healthcare "
        "professionals based on comprehensive patient evaluation."
    )
    
    st.markdown("---")
    st.caption("© 2025 Heart Failure Risk Prediction Tool")

# --- App Title  ---
st.markdown('<h1 class="main-title"><span class="heart-icon">🫀</span> Heart Failure Readmission Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict 30-day readmission risk for heart failure patients</p>', unsafe_allow_html=True)

# tabs 
tab1, tab2 = st.tabs(["🩺 Risk Assessment", "📚 User Guide"])

# --- Main Content ---
with tab1:
    st.markdown('<h2 class="section-title">📝 Patient Information</h2>', unsafe_allow_html=True)

    # Input Form 
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 class="section-title">🧑‍⚕️ Demographics</h4>', unsafe_allow_html=True)
            age = st.slider("Patient Age", 18, 100, 65)
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            ethnicity_group = st.selectbox("Ethnicity", ["Caucasian", "Non-Caucasian"])
        
        with col2:
            st.markdown('<h4 class="section-title">🩺 Clinical Information</h4>', unsafe_allow_html=True)
            icd9_category_cardiovascular = st.radio(
                "Cardiovascular Condition (ICD-9)", 
                options=[0, 1], 
                format_func=lambda x: "Present" if x == 1 else "Absent",
                horizontal=True
            )
            avg_lab_value = st.number_input("Average Lab Value", min_value=0.0, value=0.0, step=0.1)
            cpt_code_count = st.slider("Number of Procedures (CPT Codes)", 0, 20, 2)
        
        st.markdown('<h4 class="section-title">🏥 Admission Details</h4>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            admit_hour = st.slider("Admission Hour", 0, 23, 12, help="Hour of day when patient was admitted (24h format)")
            admit_day = st.selectbox("Day of Admission", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
        with col4:
            insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Government", "Self Pay"])
            admission_type = st.selectbox("Admission Type", ["EMERGENCY", "URGENT", "ELECTIVE", "NEWBORN"])
            drg_severity = st.slider("DRG Severity", 0, 4, 1, help="Diagnosis Related Group severity (0-4)")
        
        st.markdown('<br>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Generate Risk Assessment", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Feature Encoding and Prediction  ---
    if submitted:
        # progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Progress simulation
        for i in range(101):
            # Update progress bar
            progress_bar.progress(i)
            
            # Update status text based on progress
            if i < 30:
                status_text.text("🔍 Analyzing patient data...")
            elif i < 60:
                status_text.text("📊 Running predictive model...")
            elif i < 90:
                status_text.text("📋 Generating recommendations...")
            else:
                status_text.text("✅ Finalizing assessment...")
            
            time.sleep(0.01)
        
        # Prepare and encode data
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
            input_df = input_df[model.feature_names_in_]  

        # Prediction
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = int(prob >= best_threshold)
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
        # Display results card
        st.markdown('<h3 class="section-title">Risk Assessment Results</h3>', unsafe_allow_html=True)
        
        # Patient summary 

# Theme-aware metric styling
        # Theme-aware styling for metric cards
        st.markdown(f'<h5 style="color:#555;">Patient Summary</h5>', unsafe_allow_html=True)

        st.markdown(f"""
        <style>
        div[data-testid="stMetric"] {{
            background-color: {METRIC_BG};
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.05);
            text-align: center;
        }}

        div[data-testid="stMetric"] > label {{
            color: {METRIC_LABEL_COLOR};
            font-size: 2rem;
            font-weight: bold;
        }}

        div[data-testid="stMetric"] > div:nth-child(2) {{
            color: {METRIC_VALUE_COLOR};
            font-size: 1.2rem;
            
        }}
        </style>
        """, unsafe_allow_html=True)


        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        col_sum1.metric("Age", f"{age} years")
        col_sum2.metric("Gender", gender)
        col_sum3.metric("Admission", admission_type)
        col_sum4.metric("DRG Severity", f"{drg_severity}/4")


        
        # Risk visualization 
        st.markdown('<h5 style="color:#555; margin-top:15px;">Readmission Risk</h5>', unsafe_allow_html=True)
        
        # Create custom risk score visualization
        risk_percentage = int(prob * 100)
        
        if prediction:
            bar_color = "#e74c3c"
        else:
            bar_color = "#2ecc71"
        
        # Clinical assessment and recommendations 
        if prediction:
             st.markdown(f"""
            <div style="background-color:{CARD_BG}; padding: 20px; border-radius: 10px;">
                <h3 style="color:{HIGH_RISK_COLOR}; display:flex; align-items:center;">
                    <span style="margin-right:10px;">✅</span> Low Risk of Readmission
                </h3>
                <div style="display:flex; flex-wrap:wrap; gap:20px;">
                    <div style="flex:1; min-width:200px;">
                        <h4 style="color:{TEXT_COLOR};">Clinical Assessment:</h4>
                        <ul style="color:{TEXT_COLOR};">
                            <li>Patient shows low risk for 30-day readmission</li>
                            <li>Standard follow-up appropriate</li>
                        </ul>
                    </div>
                    <div style="flex:1; min-width:200px;">
                        <h4 style="color:{TEXT_COLOR};">Recommended Actions:</h4>
                        <ul style="color:{TEXT_COLOR};">
                            <li>Schedule routine follow-up (2–4 weeks)</li>
                            <li>Standard discharge instructions</li>
                            <li>Provide educational materials</li>
                            <li>Normal monitoring protocol</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:{CARD_BG}; padding: 20px; border-radius: 10px;">
                <h3 style="color:{LOW_RISK_COLOR}; display:flex; align-items:center;">
                    <span style="margin-right:10px;">✅</span> Low Risk of Readmission
                </h3>
                <div style="display:flex; flex-wrap:wrap; gap:20px;">
                    <div style="flex:1; min-width:200px;">
                        <h4 style="color:{TEXT_COLOR};">Clinical Assessment:</h4>
                        <ul style="color:{TEXT_COLOR};">
                            <li>Patient shows low risk for 30-day readmission</li>
                            <li>Standard follow-up appropriate</li>
                        </ul>
                    </div>
                    <div style="flex:1; min-width:200px;">
                        <h4 style="color:{TEXT_COLOR};">Recommended Actions:</h4>
                        <ul style="color:{TEXT_COLOR};">
                            <li>Schedule routine follow-up (2–4 weeks)</li>
                            <li>Standard discharge instructions</li>
                            <li>Provide educational materials</li>
                            <li>Normal monitoring protocol</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

        # Additional information 
        with st.expander("View Detailed Risk Factors Analysis"):
            st.markdown('<h5 style="color:#555;">Risk Factors Analysis</h5>', unsafe_allow_html=True)
            
            # Create two columns for risk factors
            factor_col1, factor_col2 = st.columns(2)
            
            with factor_col1:
                st.markdown('<h6 style="color:#0072B5;">Demographics & Timing</h6>', unsafe_allow_html=True)
                factors_demo = pd.DataFrame({
                    'Factor': ['Age', 'Gender', 'Ethnicity', 'Admission Day', 'Admission Hour'],
                    'Value': [age, gender, ethnicity_group, admit_day, f"{admit_hour}:00"]
                })
                st.dataframe(factors_demo, hide_index=True, use_container_width=True)
            
            with factor_col2:
                st.markdown('<h6 style="color:#0072B5;">Clinical & Administrative</h6>', unsafe_allow_html=True)
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
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Print date/time of assessment 
        st.markdown(f"""
        <div style="text-align:center; color:#666; font-style:italic; margin-top:10px;">
            Assessment generated on: May 11, 2025
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show placeholder 
        st.markdown("""
        <div style="background-color:#f8f9fa; border-left:4px solid #0072B5; padding:20px; border-radius:8px; 
        text-align:center; animation: fadeIn 1s ease-out;">
            <div style="font-size:40px; margin-bottom:10px;">👆</div>
            <p style="margin:0; font-size:16px;">Fill in the patient information above and click 'Generate Risk Assessment' to view results.</p>
        </div>
        """, unsafe_allow_html=True)

# User Guide Tab 
with tab2:
    st.markdown('<h2 class="section-title">📚 User Guide</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color:#0072B5;">How to Use This Tool</h3>', unsafe_allow_html=True)
    st.markdown("""
    1. **Enter Patient Data**: Fill in all required fields with the patient's information.
    2. **Submit**: Click the 'Generate Risk Assessment' button to analyze the readmission risk.
    3. **Interpret Results**: Review the risk score and recommendations for patient care planning.
    """)
    
    st.markdown('<h3 style="color:#0072B5; margin-top:20px;">Input Fields Explained</h3>', unsafe_allow_html=True)
    
    with st.expander("Patient Demographics"):
        st.markdown("""
        - **Age**: Patient's age in years
        - **Gender**: Patient's gender (Male/Female)
        - **Ethnicity**: Patient's ethnic group (Caucasian/Non-Caucasian)
        """)
        
    with st.expander("Clinical Data"):
        st.markdown("""
        - **Average Lab Value**: Mean of normalized lab values (30-300)
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
    
    st.markdown('<h3 style="color:#0072B5; margin-top:20px;">Model Information</h3>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="background-color:#f8f9fa; padding:15px; border-radius:8px; margin-top:20px; text-align:center; animation: fadeIn 1.2s ease-out;">
    <p style="color:#555; margin:0;">This tool is designed to assist healthcare providers in identifying patients who may benefit from enhanced care coordination or discharge planning.</p>
    <p style="color:#0072B5; margin-top:10px; font-weight:500;">❤️ Helping improve patient outcomes through predictive analytics</p>
</div>
""", unsafe_allow_html=True)
