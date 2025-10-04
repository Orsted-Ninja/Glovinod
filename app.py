import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Glovinod",
    page_icon="✨",
    layout="centered"
)

@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model Core '{path}' not found. System integrity compromised.")
        return None
    except Exception as e:
        st.error(f"CRITICAL FAILURE during model initialization: {e}")
        return None

MODEL_PATH = "model.joblib"
CLASS_NAMES = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
model = load_model(MODEL_PATH)


st.markdown("""
<style>
    @keyframes animated-gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(45deg, #0f0c29, #302b63, #24243e, #e60073);
        background-size: 400% 400%;
        animation: animated-gradient 15s ease infinite;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(21, 21, 21, 0.9);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #e60073;
        text-shadow: 0 0 3px #e60073;
        font-family: 'Consolas', 'Menlo', 'monospace';
        text-transform: uppercase;
    }
    .result-card {
        padding: 25px;
        border-radius: 5px;
        margin-top: 20px;
        background-color: #222222;
        border: 1px solid #e60073;
        box-shadow: 0 0 15px rgba(230, 0, 115, 0.5);
    }
    .result-card h3 {
        margin-top: 0;
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        text-shadow: none;
    }
    .result-card p {
        font-size: 16px;
        line-height: 1.6;
        color: #c0c0c0;
    }
    .stSlider .st-bf {
        background-color: #e60073;
    }
    .stButton>button {
        border: 2px solid #e60073;
        background-color: transparent;
        color: #e60073;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e60073;
        color: #151515;
        box-shadow: 0 0 20px #e60073;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; letter-spacing: 12px;'>GLOVINOD</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center;'>Planetary Analysis Core</h3>", unsafe_allow_html=True)
st.divider()

st.header("TARGET PARAMETERS")

if model is None:
    st.error("MODEL CORE OFFLINE. CANNOT ACCEPT PARAMETERS.")
else:
    input_data = {}
    st.subheader("Orbital & Planetary Metrics")
    input_data['koi_period'] = st.slider("Orbital Period (days)", 0.1, 500.0, 11.5, format="%.2f")
    input_data['koi_time0bk'] = st.slider("Transit Epoch (BKJD)", 120.0, 1500.0, 134.0)
    input_data['koi_impact'] = st.slider("Impact Parameter", 0.0, 2.0, 0.4, format="%.2f")
    input_data['koi_duration'] = st.slider("Transit Duration (hours)", 0.1, 24.0, 3.0, format="%.2f")
    input_data['koi_depth'] = st.slider("Transit Depth (ppm)", 0.0, 100000.0, 1500.0)
    input_data['koi_prad'] = st.slider("Planetary Radius (Earth radii)", 0.1, 100.0, 2.5, format="%.2f")

    st.subheader("Stellar & Flux Data")
    input_data['koi_teq'] = st.slider("Equilibrium Temperature (K)", 50.0, 5000.0, 600.0)
    input_data['koi_insol'] = st.slider("Insolation Flux (Earth flux)", 0.0, 100000.0, 200.0)
    input_data['koi_model_snr'] = st.slider("Transit Signal-to-Noise (SNR)", 0.0, 1000.0, 30.0, format="%.1f")
    input_data['koi_steff'] = st.slider("Stellar Effective Temperature (K)", 3000.0, 8000.0, 5500.0)
    input_data['koi_slogg'] = st.slider("Stellar Surface Gravity (log(g))", 2.0, 5.0, 4.4, format="%.2f")
    input_data['koi_srad'] = st.slider("Stellar Radius (Solar radii)", 0.1, 10.0, 1.0, format="%.2f")

    st.subheader("System Anomaly Flags")
    input_data['koi_fpflag_nt'] = st.selectbox("Not Transit-Like", [0, 1])
    input_data['koi_fpflag_ss'] = st.selectbox("Stellar Eclipse", [0, 1])
    input_data['koi_fpflag_co'] = st.selectbox("Centroid Offset", [0, 1])
    input_data['koi_fpflag_ec'] = st.selectbox("Ephemeris Match", [0, 1])
    
    st.write("")
    classify_button = st.button("INITIATE ANALYSIS", use_container_width=True, type="primary")

    if classify_button:
        with st.spinner('ANALYZING TARGET SIGNATURE...'):
            features_df = pd.DataFrame([input_data])
            prediction_encoded = model.predict(features_df)
            prediction_proba = model.predict_proba(features_df)
            predicted_class = CLASS_NAMES[prediction_encoded[0]]

        st.header("ANALYSIS COMPLETE")

        if predicted_class == 'CONFIRMED':
            st.markdown(
                '<div class="result-card"><h3><span style="color:#2ecc71;">✅ TARGET CONFIRMED: EXOPLANET</span></h3><p>Signal integrity is at 100%. All telemetry aligns with a confirmed planetary body. Mission success.</p></div>',
                unsafe_allow_html=True
            )
            st.balloons()
        elif predicted_class == 'CANDIDATE':
            st.markdown(
                '<div class="result-card"><h3><span style="color:#f39c12;">⚠️ TARGET DESIGNATION: CANDIDATE</span></h3><p>High-priority signal detected. Shows strong exoplanet characteristics but requires further deep-scan analysis to rule out system anomalies.</p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card"><h3><span style="color:#e74c3c;">❌ WARNING: FALSE POSITIVE</span></h3><p>System alert. Signal origin is non-planetary. Telemetry matches ghost signatures from stellar phenomena. Disregard target.</p></div>',
                unsafe_allow_html=True
            )
            
        st.subheader("CONFIDENCE MATRIX")
        proba_df = pd.DataFrame(prediction_proba, columns=CLASS_NAMES).T
        proba_df.rename(columns={0: 'Probability'}, inplace=True)
        
        color_map = {'CANDIDATE': '#f39c12', 'CONFIRMED': '#2ecc71', 'FALSE POSITIVE': '#e74c3c'}
        
        fig = go.Figure(data=[go.Bar(
            x=proba_df.index,
            y=proba_df['Probability'],
            marker_color=[color_map[cat] for cat in proba_df.index],
            text=proba_df['Probability'].apply(lambda x: f'{x:.2%}'),
            textposition='auto',
        )])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            xaxis_title="Classification",
            yaxis_title="Confidence",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
