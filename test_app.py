import pytest
import pandas as pd
import numpy as np
from app import load_model

# --- Fixtures ---
@pytest.fixture
def sample_input():
    return {
        'scaled_age': 65,
        'scaled_admit_hour': 12,
        'scaled_admit_day': 3,
        'scaled_avg_lab_value': 1.0,
        'scaled_cpt_code_count': 2,
        'scaled_drg_severity': 1,
        'gender_M': 1,
        'gender_F': 0,
        'ethnicity_group_Non-Caucasian': 0,
        'insurance_type_Medicare': 1,
        'insurance_type_Medicaid': 0,
        'insurance_type_Private': 0,
        'insurance_type_Government': 0,
        'insurance_type_Self_Pay': 0,
        'admission_type_EMERGENCY': 1,
        'admission_type_URGENT': 0,
        'admission_type_ELECTIVE': 0,
        'admission_type_NEWBORN': 0,
        'icd9_category_Cardiovascular': 1
    }

# --- Tests ---
def test_model_loading():
    model, threshold = load_model()
    assert model is not None
    assert isinstance(threshold, float)
    assert 0 <= threshold <= 1

def test_input_format(sample_input):
    df = pd.DataFrame([sample_input])
    assert df.shape == (1, 19)
    assert "scaled_age" in df.columns
    assert df["gender_M"].iloc[0] == 1

def test_prediction_shape(sample_input):
    model, _ = load_model()
    df = pd.DataFrame([sample_input])
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    prob = model.predict_proba(df)
    assert prob.shape == (1, 2)
    assert 0 <= prob[0][1] <= 1

# --- Edge Case Tests ---
def test_missing_column_handling(sample_input):
    model, _ = load_model()
    sample_input.pop('scaled_drg_severity')  # Remove one required column
    df = pd.DataFrame([sample_input])
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    prob = model.predict_proba(df)
    assert prob.shape == (1, 2)

def test_extra_column_ignored(sample_input):
    model, _ = load_model()
    sample_input['extra_feature'] = 999
    df = pd.DataFrame([sample_input])
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    prob = model.predict_proba(df)
    assert prob.shape == (1, 2)

def test_unexpected_data_types(sample_input):
    model, _ = load_model()
    sample_input['scaled_age'] = 'sixty-five'  # wrong type
    df = pd.DataFrame([sample_input])
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Check if the model silently handles the incorrect type, instead of raising an error.
    prob = model.predict_proba(df)
    
    # Validate the probabilities: 
    # - They should be a valid probability (between 0 and 1) and the sum should be 1.
    assert prob.shape == (1, 2)
    assert 0 <= prob[0][0] <= 1  # Probability for class 0
    assert 0 <= prob[0][1] <= 1  # Probability for class 1
    assert np.isclose(prob[0][0] + prob[0][1], 1)  # The sum of probabilities should be 1
