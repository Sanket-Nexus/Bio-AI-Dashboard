import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np



import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

# This decorator tells Streamlit to run this function only once and store the result
@st.cache_data
def load_model():
    """Loads the saved model and scaler from pickle files."""
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def get_sidebar_input():
    """Gets user input for all 30 features from the sidebar."""
    st.sidebar.header("Tumor Feature Input")
    st.sidebar.markdown("---")

    # The 30 features in the order the model expects them
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
        'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    # Use a dictionary to store the user's inputs
    input_dict = {}
    
    # Create a slider for each feature
    for feature in feature_names:
        # We need to set reasonable min, max, and default values for each slider
        # This is a bit manual, but necessary for a good user experience
        min_val, max_val, default_val = (0.0, 50.0, 15.0) # Example default values
        
        # You can get more accurate default ranges from your dataframe's .describe() method
        # For simplicity, we'll use a generic range for this tutorial
        
        input_dict[feature] = st.sidebar.slider(
            label=f"{feature.replace('_', ' ').title()}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.1
        )
        
    # Convert the dictionary of inputs into a Pandas DataFrame
    return pd.DataFrame([input_dict])

def main():
    # --- APP LAYOUT ---
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="♋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load the model and scaler
    model, scaler = load_model()

    # --- MAIN INTERFACE ---
    st.title("Breast Cancer Prediction App 🔬")
    st.write(
        "This app uses a Logistic Regression model to predict whether a breast tumor is malignant or benign. "
        "Please use the sliders in the sidebar to enter the tumor's measurements."
    )
    
    # Get user input from the sidebar
    input_df = get_sidebar_input()

    # --- PREDICTION LOGIC ---
    if st.sidebar.button("Predict"):
        # Scale the user's input data
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # --- DISPLAY RESULTS ---
        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.success("The tumor is **Benign**.")
            st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
        else:
            st.error("The tumor is **Malignant**.")
            st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")

# This is the standard way to run a Python script
if __name__ == '__main__':
    main()









