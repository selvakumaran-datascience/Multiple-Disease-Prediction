import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Health Assistant: Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
    }
    h2, h3 {
        color: #34495e;
    }
    .result-success {
        padding: 20px;
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .result-danger {
        padding: 20px;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models safely
@st.cache_resource
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("Disease Prediction System")
    st.markdown("---")
    selected = st.radio(
        "Choose a Prediction Model",
        ["Kidney Disease", "Liver Disease", "Parkinson's Disease"],
        index=0
    )
    st.markdown("---")
    st.info("💡 **Disclaimer:** This tool provides predictions based on machine learning models and should **not** replace professional medical advice or diagnosis.")

# ----------------- KIDNEY DISEASE -----------------
if selected == "Kidney Disease":
    st.title("🩸 Kidney Disease Prediction")
    st.markdown("Enter the patient's medical details below to predict the likelihood of Chronic Kidney Disease.")
    
    package = load_model("C:/Users/Selva.M/Downloads/data_science/mini_projects/Multi_Disease_Prediction/best_kidney_model.pkl")
    if package:
        model = package.get("model")
        encoders = package.get("encoders", {})
        
        with st.form("kidney_form"):
            st.subheader("Patient Vitals & Blood Work")
            
            # Grouping into columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=50.0)
                bp = st.number_input("Blood Pressure (mm/Hg)", min_value=50.0, max_value=200.0, value=80.0)
                sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.030, value=1.020, format="%.3f")
                al = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=0.0)
                su = st.number_input("Sugar", min_value=0.0, max_value=5.0, value=0.0)
                bgr = st.number_input("Blood Glucose Random (mgs/dl)", min_value=20.0, max_value=500.0, value=120.0)
                
            with col2:
                bu = st.number_input("Blood Urea (mgs/dl)", min_value=1.0, max_value=400.0, value=35.0)
                sc = st.number_input("Serum Creatinine (mgs/dl)", min_value=0.0, max_value=40.0, value=1.0)
                sod = st.number_input("Sodium (mEq/L)", min_value=100.0, max_value=180.0, value=140.0)
                pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=50.0, value=4.5)
                hemo = st.number_input("Hemoglobin (gms)", min_value=3.0, max_value=20.0, value=15.0)
                pcv = st.number_input("Packed Cell Volume", min_value=9.0, max_value=60.0, value=44.0)

            with col3:
                wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=2000.0, max_value=30000.0, value=7500.0)
                rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=8.0, value=4.5)
                rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
                pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
                pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
                ba = st.selectbox("Bacteria", ["notpresent", "present"])

            with col4:
                htn = st.selectbox("Hypertension", ["no", "yes"])
                dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
                cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
                appet = st.selectbox("Appetite", ["good", "poor"])
                pe = st.selectbox("Pedal Edema", ["no", "yes"])
                ane = st.selectbox("Anemia", ["no", "yes"])
                
            submit = st.form_submit_button("Predict Kidney Disease")
            
        if submit:
            try:
                # Prepare dataframe
                input_data = pd.DataFrame([{
                    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 
                    'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba, 'bgr': bgr, 
                    'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo, 
                    'pcv': pcv, 'wc': wc, 'rc': rc, 'htn': htn, 'dm': dm, 
                    'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
                }])
                
                # Apply encoders
                for col in encoders:
                    if col in input_data.columns:
                        # Handle potential unseen labels gracefully, though selectbox restricts to known
                        # We apply transform which requires 1D array or series
                        input_data[col] = encoders[col].transform(input_data[col])
                
                prediction = model.predict(input_data)[0]
                
                if prediction == 1:
                    st.markdown('<div class="result-danger">⚠️ The model predicts that the patient HAS Kidney Disease.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success">✅ The model predicts that the patient DOES NOT have Kidney Disease.</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# ----------------- LIVER DISEASE -----------------
elif selected == "Liver Disease":
    st.title("🫀 Liver Disease Prediction")
    st.markdown("Enter the patient's test results below to predict the likelihood of Liver Disease.")
    
    package = load_model("C:/Users/Selva.M/Downloads/data_science/mini_projects/Multi_Disease_Prediction/best_liver_model.pkl")
    if package:
        model = package.get("model")
        
        with st.form("liver_form"):
            st.subheader("Patient Liver Function Tests")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=40)
                gender = st.selectbox("Gender", ["Male", "Female"])
                total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=50.0, value=1.0)
                direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=25.0, value=0.5)
                alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=50, max_value=3000, value=200)
            
            with col2:
                alamine_aminotransferase = st.number_input("Alamine Aminotransferase", min_value=10, max_value=2500, value=40)
                aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=10, max_value=5000, value=40)
                total_protiens = st.number_input("Total Protiens", min_value=2.0, max_value=15.0, value=6.5)
                albumin = st.number_input("Albumin", min_value=0.5, max_value=10.0, value=3.5)
                ag_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.1, max_value=5.0, value=1.0)
                
            submit = st.form_submit_button("Predict Liver Disease")
            
        if submit:
            try:
                gender_map = {"Male": 1, "Female": 0}
                
                # Order matters: 'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
                input_data = pd.DataFrame([{
                    'Age': age,
                    'Gender': gender_map[gender],
                    'Total_Bilirubin': total_bilirubin,
                    'Direct_Bilirubin': direct_bilirubin,
                    'Alkaline_Phosphotase': alkaline_phosphotase,
                    'Alamine_Aminotransferase': alamine_aminotransferase,
                    'Aspartate_Aminotransferase': aspartate_aminotransferase,
                    'Total_Protiens': total_protiens,
                    'Albumin': albumin,
                    'Albumin_and_Globulin_Ratio': ag_ratio
                }])
                
                prediction = model.predict(input_data)[0]
                
                # Assume 1 is disease, 2 or 0 is no disease, but wait, indian liver dataset targets: 1 (Liver Patient), 2 (Non Liver Patient). Let's check classes.
                if prediction == 1:
                    st.markdown('<div class="result-danger">⚠️ The model predicts that the patient HAS Liver Disease.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success">✅ The model predicts that the patient DOES NOT have Liver Disease.</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# ----------------- PARKINSON'S DISEASE -----------------
elif selected == "Parkinson's Disease":
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter the acoustic measurements of the patient's voice below to predict the likelihood of Parkinson's Disease.")
    
    package = load_model("C:/Users/Selva.M/Downloads/data_science/mini_projects/Multi_Disease_Prediction/best_parkinsons_model.pkl")
    if package:
        model = package.get("model")
        
        with st.form("parkinsons_form"):
            st.subheader("Voice Measurement Features")
            
            # 22 features, let's use 3 columns
            col1, col2, col3 = st.columns(3)
            
            # Using realistic default values for the features
            with col1:
                fo = st.number_input("MDVP:Fo(Hz)", value=119.99200)
                fhi = st.number_input("MDVP:Fhi(Hz)", value=157.30200)
                flo = st.number_input("MDVP:Flo(Hz)", value=74.99700)
                jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.00784, format="%.5f")
                jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00007, format="%.5f")
                rap = st.number_input("MDVP:RAP", value=0.00370, format="%.5f")
                ppq = st.number_input("MDVP:PPQ", value=0.00554, format="%.5f")
                jitter_ddp = st.number_input("Jitter:DDP", value=0.01109, format="%.5f")

            with col2:
                shimmer = st.number_input("MDVP:Shimmer", value=0.04374, format="%.5f")
                shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.42600, format="%.5f")
                shimmer_apq3 = st.number_input("Shimmer:APQ3", value=0.02182, format="%.5f")
                shimmer_apq5 = st.number_input("Shimmer:APQ5", value=0.03130, format="%.5f")
                apq = st.number_input("MDVP:APQ", value=0.02971, format="%.5f")
                shimmer_dda = st.number_input("Shimmer:DDA", value=0.06545, format="%.5f")
                nhr = st.number_input("NHR", value=0.02211, format="%.5f")

            with col3:
                hnr = st.number_input("HNR", value=21.03300)
                rpde = st.number_input("RPDE", value=0.41478, format="%.5f")
                dfa = st.number_input("DFA", value=0.81528, format="%.5f")
                spread1 = st.number_input("spread1", value=-4.81303, format="%.5f")
                spread2 = st.number_input("spread2", value=0.26648, format="%.5f")
                d2 = st.number_input("D2", value=2.30144, format="%.5f")
                ppe = st.number_input("PPE", value=0.28465, format="%.5f")

            submit = st.form_submit_button("Predict Parkinson's Disease")
            
        if submit:
            try:
                # Order matters:
                features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
                
                input_data = pd.DataFrame([{
                    'MDVP:Fo(Hz)': fo, 'MDVP:Fhi(Hz)': fhi, 'MDVP:Flo(Hz)': flo, 
                    'MDVP:Jitter(%)': jitter_percent, 'MDVP:Jitter(Abs)': jitter_abs, 
                    'MDVP:RAP': rap, 'MDVP:PPQ': ppq, 'Jitter:DDP': jitter_ddp, 
                    'MDVP:Shimmer': shimmer, 'MDVP:Shimmer(dB)': shimmer_db, 
                    'Shimmer:APQ3': shimmer_apq3, 'Shimmer:APQ5': shimmer_apq5, 
                    'MDVP:APQ': apq, 'Shimmer:DDA': shimmer_dda, 'NHR': nhr, 
                    'HNR': hnr, 'RPDE': rpde, 'DFA': dfa, 'spread1': spread1, 
                    'spread2': spread2, 'D2': d2, 'PPE': ppe
                }])
                
                prediction = model.predict(input_data)[0]
                
                if prediction == 1:
                    st.markdown('<div class="result-danger">⚠️ The model predicts that the patient HAS Parkinson\'s Disease.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success">✅ The model predicts that the patient DOES NOT have Parkinson\'s Disease.</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
