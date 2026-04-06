import streamlit as st
import requests
from PIL import Image
import numpy as np
import os
import pandas as pd
import base64
from io import BytesIO
from datetime import datetime

# Import Custom Modules
from utils import database
from utils import report_generator

# --- Configuration ---
st.set_page_config(page_title="PancreScan AI", page_icon="🏥", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = os.environ.get("API_URL", "http://api:8000")

# --- Custom CSS ---
ST_STYLE = """
<style>
    /* General Settings */
    html, body, .stApp {
        font-family: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }

    /* Headings */
    h1, h2, h3 {
        color: #0056b3; /* Deep Blue */
        font-weight: 700;
    }
    h1 {
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 15px;
        margin-bottom: 30px;
    }
    
    /* Hero/Header */
    .hero-container {
        background: linear-gradient(135deg, #0056b3 0%, #663399 100%);
        padding: 40px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 10px;
        color: white !important;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }

    /* Result Cards */
    .result-card {
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }
    .result-card-normal {
        background-color: #ffffff;
        border-left: 6px solid #28a745;
    }
    .result-card-tumor {
        background-color: #ffffff;
        border-left: 6px solid #dc3545;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .normal-text { color: #28a745; }
    .tumor-text { color: #dc3545; }
    
    .confidence-score {
        font-size: 2rem;
        font-weight: 800;
        color: #343a40;
    }
    .confidence-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Stats Section */
    .stats-container {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-top: 40px;
    }
    .stat-metric {
        text-align: center;
        padding: 20px;
        border-right: 1px solid #e9ecef;
    }
    .stat-metric:last-child {
        border-right: none;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #663399; /* Purple for Pancreatic Cancer Awareness */
    }
    .stat-label {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 600;
    }
    .source-text {
        font-size: 0.8rem;
        color: #adb5bd;
        text-align: right;
        margin-top: 10px;
        font-style: italic;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f1f3f5;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0056b3;
        color: white;
        border-radius: 6px;
        padding: 12px 28px;
        border: none;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #004494;
        color: white;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

</style>
"""

# --- App Layout ---
st.markdown(ST_STYLE, unsafe_allow_html=True)


# --- Helper to Call API ---
@st.cache_data(show_spinner=False)
def analyze_scan(image_bytes, model_name, threshold):
    files = {"file": ("scan.jpg", image_bytes, "image/jpeg")}
    params = {"heatmap": True, "threshold": threshold, "model_name": model_name}
    
    try:
        response = requests.post(f"{API_URL}/predict", files=files, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        return None

def render_hero_section():
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">PancreScan AI</div>
            <div class="hero-subtitle">Advanced Deep Learning for Early Pancreatic Tumor Detection</div>
        </div>
    """, unsafe_allow_html=True)

def render_stats_section():
    st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
    st.markdown("## 📊 2024 Pancreatic Cancer Landscape", unsafe_allow_html=True)
    st.markdown("Key statistics highlighting the importance of early detection.", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="stat-metric">
                <div class="stat-value">66,440</div>
                <div class="stat-label">Estimated New Cases (2024)</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="stat-metric">
                <div class="stat-value">51,750</div>
                <div class="stat-label">Estimated Deaths (2024)</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="stat-metric">
                <div class="stat-value">13%</div>
                <div class="stat-label">5-Year Survival Rate</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Survival Rate by Stage")
        # Data for survival rates
        stages = ['Localized', 'Regional', 'Distant', 'All Stages']
        rates = [44, 16, 3, 13] # % survival
        
        # Simple bar chart using native streamlit for professionalism and speed
        chart_data = {"Stage": stages, "5-Year Survival (%)": rates}
        st.bar_chart(chart_data, x="Stage", y="5-Year Survival (%)", color="#663399")
        st.caption("Survival rates drop drastically significantly when cancer spreads only to nearby structures (Regional) or distant organs (Distant). Early detection (Localized) offers the best chance. Source: SEER (2024).")

    with c2:
        st.subheader("Estimated New Cases by Age")
        # Approximate distribution data
        age_groups = ['Under 45', '45-54', '55-64', '65-74', '75+']
        counts = [2, 6, 20, 28, 44] # Approximate percentages
        
        st.bar_chart({"Age Group": age_groups, "% of Diagnoses": counts}, x="Age Group", y="% of Diagnoses", color="#0056b3")
        st.caption("Risk increases significantly with age, with the majority of patients diagnosed after age 65. Source: National Cancer Institute.")

    st.markdown('<div class="source-text">Data Sources: American Cancer Society, SEER Database, 2024 Statistics.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("🏥 PancreScan AI")
    st.sidebar.markdown("---")
    
    # Mode Selection
    mode = st.sidebar.radio("Navigation", ["Single Scan", "Batch Analysis", "Patient History"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    model_selector = st.sidebar.selectbox(
        "Model Architecture",
        ["EfficientNet-V2-S (Recommended)", "DenseNet121", "ConvNeXt-Tiny", "UNet (Multi-Task Segmentation)"]
    )
    
    # --- Model Performance Metrics (from 5-Fold Cross-Validation) ---
    MODEL_METRICS = {
        "EfficientNet-V2-S (Recommended)": {
            "accuracy": 98.50, "f1": 98.46, "precision": 98.52, "recall": 98.96,
            "folds": 5, "epochs": 20
        },
        "DenseNet121": {
            "accuracy": 98.20, "f1": 98.17, "precision": 97.97, "recall": 96.89,
            "folds": 5, "epochs": 20
        },
        "ConvNeXt-Tiny": {
            "accuracy": 98.30, "f1": 98.26, "precision": 98.15, "recall": 97.75,
            "folds": 5, "epochs": 20
        },
    }
    
    metrics = MODEL_METRICS.get(model_selector, {})
    if metrics:
        st.sidebar.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0056b3 0%, #663399 100%);
                        padding: 15px; border-radius: 10px; margin-top: 10px; color: white;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8; margin-bottom: 8px;">
                    📊 Model Performance
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 8px; padding: 10px;">
                        <div style="font-size: 1.4rem; font-weight: 800;">{metrics['accuracy']:.1f}%</div>
                        <div style="font-size: 0.7rem; opacity: 0.9;">Accuracy</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 8px; padding: 10px;">
                        <div style="font-size: 1.4rem; font-weight: 800;">{metrics['f1']:.1f}%</div>
                        <div style="font-size: 0.7rem; opacity: 0.9;">F1 Score</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 8px; padding: 10px;">
                        <div style="font-size: 1.4rem; font-weight: 800;">{metrics['precision']:.1f}%</div>
                        <div style="font-size: 0.7rem; opacity: 0.9;">Precision</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 8px; padding: 10px;">
                        <div style="font-size: 1.4rem; font-weight: 800;">{metrics['recall']:.1f}%</div>
                        <div style="font-size: 0.7rem; opacity: 0.9;">Recall</div>
                    </div>
                </div>
                <div style="font-size: 0.65rem; opacity: 0.6; margin-top: 8px; text-align: right;">
                    {metrics['folds']}-Fold CV · {metrics['epochs']} Epochs
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.4, 0.05, help="Lower threshold increases sensitivity to potential tumors.")
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About**\n\n"
        "PancreScan uses deep learning to assist radiologists in detecting pancreatic tumors from CT slices.\n\n"
        "© 2026 PancreScan Team"
    )

    # Main Area
    render_hero_section()
    
    if mode == "Single Scan":
        render_single_scan_ui(model_selector, threshold)
    elif mode == "Batch Analysis":
        render_batch_analysis_ui(model_selector, threshold)
    elif mode == "Patient History":
        render_patient_history_ui()
    
    # Render Statistics at the bottom
    st.markdown("---")
    render_stats_section()

def render_single_scan_ui(model_selector, threshold):
    st.subheader("🔎 Single Scan Analysis")
    
    # Patient Association (Optional)
    patients = database.get_all_patients()
    patient_options = ["None (Anonymous)"] + [f"{p['name']} ({p['mrn']})" for p in patients]
    selected_patient_str = st.selectbox("Assign to Patient (Optional)", patient_options)
    
    selected_patient_id = None
    if selected_patient_str != "None (Anonymous)":
        mrn = selected_patient_str.split('(')[-1].strip(')')
        patient = database.get_patient(mrn)
        if patient:
            selected_patient_id = patient['id']

    # Input Tabs
    tab1, tab2 = st.tabs(["📂 Upload Scan", "🖼️ Try Example"])
    
    image = None
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a CT Slice (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            # Only reset if we upload a new file, but file uploader state persists
            if st.session_state.get('last_uploaded_file') != uploaded_file.name:
                st.session_state['current_image'] = Image.open(uploaded_file).convert("RGB")
                st.session_state['last_uploaded_file'] = uploaded_file.name
                st.session_state['analyzed'] = False
                st.session_state.pop('tumor_prob', None)
                st.session_state.pop('last_mask', None)
            
    with tab2:
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("Load Normal Example"):
                image_path = os.path.join(BASE_DIR, "DATASET", "test", "test", "normal", "1-001.jpg")
                if os.path.exists(image_path):
                    st.session_state['current_image'] = Image.open(image_path).convert("RGB")
                    st.session_state['analyzed'] = False
                    st.session_state.pop('tumor_prob', None)
                    st.session_state.pop('last_mask', None)
                else:
                    st.error("Example image not found.")
        with col_ex2:
            if st.button("Load Tumor Example"):
                image_path = os.path.join(BASE_DIR, "DATASET", "test", "test", "pancreatic_tumor", "1-001.jpg")
                if os.path.exists(image_path):
                    st.session_state['current_image'] = Image.open(image_path).convert("RGB")
                    st.session_state['analyzed'] = False
                    st.session_state.pop('tumor_prob', None)
                    st.session_state.pop('last_mask', None)
                else:
                    st.error("Example image not found.")
    
    # Get image from session
    image = st.session_state.get('current_image')
    
    # Analysis UI
    if image:
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.image(image, caption="Input Scan", use_container_width=True)
            
            if st.button("🔍 Run Analysis", use_container_width=True):
                with st.spinner("Analyzing scan with Ensemble APIs..."):
                    # Convert PIL Image to Bytes for API
                    img_bytes = BytesIO()
                    image.save(img_bytes, format="JPEG")
                    image_bytes_val = img_bytes.getvalue()
                    
                    api_result = analyze_scan(image_bytes_val, model_selector, threshold)
                    
                    if api_result:
                        tumor_prob = api_result.get("confidence", 0.0)
                        diagnosis = api_result.get("diagnosis", "Normal")
                        
                        # Store results in session state
                        st.session_state['tumor_prob'] = tumor_prob
                        st.session_state['analyzed'] = True
                        st.session_state['api_result'] = api_result
                        
                        # Save to DB if patient selected
                        if selected_patient_id:
                            filename = uploaded_file.name if uploaded_file else "example.jpg"
                            pred_label = "Tumor" if diagnosis == "pancreatic_tumor" else diagnosis
                            try:
                                database.add_scan(
                                    selected_patient_id, 
                                    filename, 
                                    pred_label, 
                                    tumor_prob, 
                                    model_selector
                                )
                                st.success(f"Scan saved to patient record!")
                            except Exception as e:
                                st.error(f"Failed to save to DB: {e}")
        
        with col2:
            if st.session_state.get('analyzed', False):
                api_result = st.session_state.get('api_result', {})
                tumor_prob = api_result.get("confidence", 0.0)
                diagnosis = api_result.get("diagnosis", "Unknown")
                heatmap_b64 = api_result.get("heatmap_b64")
                mask_b64 = api_result.get("mask_b64")
                
                if diagnosis == "Inconclusive":
                    st.markdown(
                        f"""
                        <div class="result-card" style="background-color: #fff3cd; border-left: 6px solid #ffcc00; color: #856404;">
                            <div class="card-title" style="color: #856404;">
                                ⚠️ Inconclusive Data
                            </div>
                            <p style="font-size: 1.1rem;">
                                The model could not find a clear pancreas region or the CT slice is poor quality.
                            </p>
                            <div style="margin-top: 15px;">
                                <div class="confidence-label">Measured Confidence</div>
                                <div class="confidence-score">{tumor_prob*100:.2f}%</div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif diagnosis == "pancreatic_tumor" or tumor_prob > threshold:
                    st.markdown(
                        f"""
                        <div class="result-card result-card-tumor">
                            <div class="card-title tumor-text">
                                🚨 Tumor Detected
                            </div>
                            <p style="font-size: 1.1rem;">The model has identified patterns consistent with pancreatic malignancy.</p>
                            <div style="margin-top: 15px;">
                                <div class="confidence-label">Confidence Score</div>
                                <div class="confidence-score">{tumor_prob*100:.2f}%</div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card result-card-normal">
                            <div class="card-title normal-text">
                                ✅ Normal Patterns
                            </div>
                            <p style="font-size: 1.1rem;">No malignant patterns detected above the sensitivity threshold.</p>
                            <div style="margin-top: 15px;">
                                <div class="confidence-label">Confidence Score</div>
                                <div class="confidence-score">{(1-tumor_prob)*100:.2f}%</div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                st.markdown("#### Probability Distribution")
                st.progress(tumor_prob)
                st.caption(f"Tumor Probability: {tumor_prob:.4f}")
                
                if mask_b64:
                    st.markdown("#### Structural Mask Overlay (Pancreas)")
                    img_bytes = base64.b64decode(mask_b64)
                    overlay_img = Image.open(BytesIO(img_bytes))
                    st.image(overlay_img, caption="Blue/Red highlight shows dynamically segmented pancreas", use_container_width=True)
                
                with st.expander("Show AI Reasoning (Grad-CAM)"):
                    if heatmap_b64:
                        img_bytes = base64.b64decode(heatmap_b64)
                        heatmap_img = Image.open(BytesIO(img_bytes))
                        st.image(heatmap_img, caption="Heatmap: Red areas indicate regions contributing to Tumor classification", use_container_width=True)
                    else:
                        st.info("No Grad-CAM available for this prediction (usually only generated for Tumor class).")

def render_batch_analysis_ui(model_selector, threshold):
    st.subheader("📦 Batch Analysis")
    
    uploaded_files = st.file_uploader("Upload Multiple CT Slices", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button(f"Analyze {len(uploaded_files)} Scans"):
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                try:
                    img_bytes = file.getvalue()
                    api_result = analyze_scan(img_bytes, model_selector, threshold)
                    
                    if api_result:
                        tumor_prob = api_result.get("confidence", 0.0)
                        diag = api_result.get("diagnosis", "Unknown")
                        results.append({
                            "Filename": file.name,
                            "Prediction": "Tumor" if diag == "pancreatic_tumor" else diag,
                            "Confidence": tumor_prob,
                            "Status": "⚠️ High Risk" if tumor_prob > threshold else ("✅ Normal" if diag != "Inconclusive" else "⚠️ Inconclusive")
                        })
                    else:
                        raise Exception("API returned empty result")
                except Exception as e:
                    results.append({
                        "Filename": file.name,
                        "Prediction": "Error",
                        "Confidence": 0.0,
                        "Status": f"Failed: {str(e)}"
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display Results
            df = pd.DataFrame(results)
            st.dataframe(df.style.map(lambda x: 'color: red' if 'High Risk' in str(x) else 'color: green', subset=['Status']), use_container_width=True)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results CSV",
                csv,
                "batch_results.csv",
                "text/csv",
                key='download-csv'
            )

from io import BytesIO

def render_patient_history_ui():
    st.subheader("📋 Patient History & Reports")
    
    tab1, tab2 = st.tabs(["Register New Patient", "View Patient Records"])
    
    with tab1:
        with st.form("new_patient_form"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Full Name")
            mrn = c2.text_input("Medical Record Number (MRN)")
            age = c1.number_input("Age", min_value=0, max_value=120)
            gender = c2.selectbox("Gender", ["Male", "Female", "Other"])
            
            if st.form_submit_button("Register Patient"):
                if name and mrn:
                    pid, msg = database.add_patient(mrn, name, age, gender)
                    if pid:
                        st.success(f"Patient registered successfully! (ID: {pid})")
                    else:
                        st.error(f"Error: {msg}")
                else:
                    st.warning("Please fill in Name and MRN.")
    
    with tab2:
        patients = database.get_all_patients()
        if not patients:
            st.info("No patients registered yet.")
        else:
            patient_options = {f"{p['name']} ({p['mrn']})": p for p in patients}
            selected_p = st.selectbox("Select Patient", list(patient_options.keys()))
            
            if selected_p:
                p_data = patient_options[selected_p]
                st.write(f"**Age:** {p_data['age']} | **Gender:** {p_data['gender']} | **Registered:** {p_data['created_at']}")
                
                history = database.get_patient_history(p_data['id'])
                
                if history:
                    st.markdown("### Scan History")
                    for scan in history:
                        with st.expander(f"{scan['scan_date']} - {scan['prediction']} ({scan['filename']})"):
                            c1, c2 = st.columns([2, 1])
                            c1.metric("Prediction", scan['prediction'])
                            c1.metric("Confidence", f"{scan['confidence']:.2%}")
                            c1.text(f"Model: {scan['model_used']}")
                            
                            # Generate PDF in memory
                            scan_results = {
                                "prediction": scan['prediction'],
                                "confidence": scan['confidence'],
                                "model": scan['model_used']
                            }
                            buffer = BytesIO()
                            report_generator.generate_report(p_data, scan_results, buffer)
                            pdf_bytes = buffer.getvalue()
                            
                            c2.download_button(
                                label="📄 Download Report",
                                data=pdf_bytes,
                                file_name=f"report_{p_data['mrn']}_{scan['id']}.pdf",
                                mime="application/pdf"
                            )
                else:
                    st.info("No scan history found for this patient.")

if __name__ == "__main__":
    main()
