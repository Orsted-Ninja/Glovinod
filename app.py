import streamlit as st
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "model.joblib" 

CLASS_NAMES = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'] 

@st.cache_resource
def load_model(path):
    """Loads the pre-trained model pipeline."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

st.set_page_config(page_title="Exoplanet Classifier", layout="wide")


st.title("🪐 Exoplanet Candidate Classifier")
st.write("""
This app uses a **Random Forest Machine Learning model** to classify potential exoplanets from transit survey data, based on the Kepler mission dataset.
Enter the observational data in the sidebar to get a prediction.
""")

st.sidebar.header("Input Observational Data")

if model is None:
   
    st.info("Please download the `exoplanet_model_pipeline.joblib` file and place it in the same folder as this script (`app.py`).")
else:
    # Initialize input data dictionary
    input_data = {}
    
    # --- Numerical Feature Sliders ---
    st.sidebar.subheader("Planetary/Orbital Parameters")
    input_data['koi_period'] = st.sidebar.slider("Orbital Period (days)", 0.1, 500.0, 11.5, format="%.2f")
    input_data['koi_time0bk'] = st.sidebar.slider("Transit Epoch (BKJD)", 120.0, 1500.0, 134.0)
    input_data['koi_impact'] = st.sidebar.slider("Impact Parameter", 0.0, 2.0, 0.4, format="%.2f")
    input_data['koi_duration'] = st.sidebar.slider("Transit Duration (hours)", 0.1, 24.0, 3.0, format="%.2f")
    input_data['koi_depth'] = st.sidebar.slider("Transit Depth (ppm)", 0.0, 100000.0, 1500.0)
    input_data['koi_prad'] = st.sidebar.slider("Planetary Radius (Earth radii)", 0.1, 100.0, 2.5, format="%.2f")
    
    st.sidebar.subheader("Stellar/Flux Parameters")
    input_data['koi_teq'] = st.sidebar.slider("Equilibrium Temperature (K)", 50.0, 5000.0, 600.0)
    input_data['koi_insol'] = st.sidebar.slider("Insolation Flux (Earth flux)", 0.0, 100000.0, 200.0)
    input_data['koi_model_snr'] = st.sidebar.slider("Transit Signal-to-Noise (SNR)", 0.0, 1000.0, 30.0, format="%.1f")
    input_data['koi_steff'] = st.sidebar.slider("Stellar Effective Temperature (K)", 3000.0, 8000.0, 5500.0)
    input_data['koi_slogg'] = st.sidebar.slider("Stellar Surface Gravity (log(g))", 2.0, 5.0, 4.4, format="%.2f")
    input_data['koi_srad'] = st.sidebar.slider("Stellar Radius (Solar radii)", 0.1, 10.0, 1.0, format="%.2f")

   
    st.sidebar.subheader("False Positive Flags (1=Flagged, 0=Not Flagged)")
    # The key names must match the feature list used for training exactly!
    input_data['koi_fpflag_nt'] = st.sidebar.selectbox("Not Transit-Like Flag (`koi_fpflag_nt`)", [0, 1])
    input_data['koi_fpflag_ss'] = st.sidebar.selectbox("Stellar Eclipse Flag (`koi_fpflag_ss`)", [0, 1])
    input_data['koi_fpflag_co'] = st.sidebar.selectbox("Centroid Offset Flag (`koi_fpflag_co`)", [0, 1])
    input_data['koi_fpflag_ec'] = st.sidebar.selectbox("Ephemeris Match Flag (`koi_fpflag_ec`)", [0, 1])

    
    if st.sidebar.button("Classify Exoplanet"):
        # Create DataFrame with the exact column order used in training
        features_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction_encoded = model.predict(features_df)
        prediction_proba = model.predict_proba(features_df)
        
        # Get the human-readable class name
        predicted_class = CLASS_NAMES[prediction_encoded[0]]
        
        # --- Display Results ---
        st.subheader("Prediction Result")
        
        if predicted_class == 'CONFIRMED':
            st.success(f"**Classification:** {predicted_class} ✅ (Model Index: 1)")
            st.markdown("The model predicts this is a **confirmed exoplanet**. This means the signal is strong and consistent with a planetary body transiting its star.")
        elif predicted_class == 'CANDIDATE':
            st.info(f"**Classification:** {predicted_class} ⚠️ (Model Index: 0)") 
            st.markdown("The model classifies this as a **planetary candidate**. The signal shows promise but may require further observation or vetting to be fully confirmed.")
        else: # FALSE POSITIVE
            st.error(f"**Classification:** {predicted_class} ❌ (Model Index: 2)")
            st.markdown("The model predicts this is a **false positive**. The signal is likely caused by other phenomena, such as an eclipsing binary star system.")

        # Display prediction probabilities
        st.subheader("Prediction Confidence (Probability by Class)")
        proba_df = pd.DataFrame(prediction_proba, columns=CLASS_NAMES).T.rename(columns={0: 'Probability'})
        st.bar_chart(proba_df)

# --- How to Run the App ---
st.markdown("---")
st.info("""
**To run this app:**
1. **Download** the generated `exoplanet_model_pipeline.joblib` file and place it in the same directory as this code.
2. **Save** the code above as `app.py`.
3. **Install** required libraries: `pip install streamlit pandas joblib scikit-learn`.
4. **Run** the app from your terminal: `streamlit run app.py`
""")