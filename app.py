import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from datetime import datetime

# Import Custom Modules
import database
import report_generator

# --- Configuration ---
st.set_page_config(page_title="PancreScan AI", page_icon="🏥", layout="wide")

MODEL_PATH_DENSE = "densenet121_best.pt"
MODEL_PATH_EFFICIENT = "outputs/demo_models/efficientnet_v2_s_fold_1_best.pt"
MODEL_PATH_CONVNEXT = "convnext_tiny_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# --- Transformations ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# --- Model Loading ---
@st.cache_resource
def load_model(model_name):
    try:
        model = None
        trained_model_path = ""
        
        if "DenseNet121" in model_name:
            model = models.densenet121(weights=None)
            trained_model_path = MODEL_PATH_DENSE
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2), 
                nn.Linear(num_ftrs, 2)
            )
        elif "EfficientNet-V2-S" in model_name:
            model = models.efficientnet_v2_s(weights=None)
            trained_model_path = MODEL_PATH_EFFICIENT
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 2)
            )
        elif "ConvNeXt-Tiny" in model_name:
            model = models.convnext_tiny(weights=None)
            trained_model_path = MODEL_PATH_CONVNEXT
            num_ftrs = model.classifier[2].in_features
            model.classifier[2] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 2)
            )
        
        if os.path.exists(trained_model_path):
            state_dict = torch.load(trained_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
        else:
            # Fallback for demo if precise weights missing
            pass 
        
        if model:
            model.to(DEVICE)
            model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Grad-CAM Helper ---
def get_target_layer(model, model_name):
    if "DenseNet" in model_name:
        return [model.features[-1]]
    elif "EfficientNet" in model_name:
        return [model.features[-1]]
    elif "ConvNeXt" in model_name:
        return [model.features[-1]]
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
        ["EfficientNet-V2-S (Recommended)", "DenseNet121", "ConvNeXt-Tiny"]
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
            image = Image.open(uploaded_file).convert("RGB")
            
    with tab2:
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("Load Normal Example"):
                image_path = "DATASET/test/test/normal/1-001.jpg" 
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                else:
                    st.error("Example image not found.")
        with col_ex2:
            if st.button("Load Tumor Example"):
                image_path = "DATASET/test/test/pancreatic_tumor/1-001.jpg"
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                else:
                    st.error("Example image not found.")
    
    # Analysis UI
    if image:
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.image(image, caption="Input Scan", use_column_width=True)
            
            if st.button("🔍 Run Analysis", use_container_width=True):
                with st.spinner("Analyzing scan pattern..."):
                    model = load_model(model_selector)
                    
                    if model:
                        # Preprocess
                        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
                        
                        # Inference
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.softmax(outputs, dim=1)
                            tumor_prob = probs[0][1].item()
                            
                        # Store results in session state
                        st.session_state['tumor_prob'] = tumor_prob
                        st.session_state['analyzed'] = True
                        
                        # Save to DB if patient selected
                        if selected_patient_id:
                            filename = uploaded_file.name if uploaded_file else "example.jpg"
                            pred_label = "Tumor" if tumor_prob > threshold else "Normal"
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
                tumor_prob = st.session_state.get('tumor_prob', 0.0)
                is_tumor = tumor_prob > threshold
                
                # Dynamic Card Rendering
                if is_tumor:
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
                
                with st.expander("Show AI Reasoning (Grad-CAM)"):
                    try:
                        target_layers = get_target_layer(model, model_selector)
                        if target_layers:
                            cam = GradCAM(model=model, target_layers=target_layers)
                            targets = [ClassifierOutputTarget(1)] # Focus on Tumor
                            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
                            
                            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                            grayscale_cam = grayscale_cam[0, :]
                            
                            img_np = np.array(image.resize((224, 224))) / 255.0
                            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                            
                            st.image(visualization, caption="Heatmap: Red areas indicate regions contributing to Tumor classification", use_column_width=True)
                    except Exception as e:
                        st.warning(f"Visualization unavailable: {e}")

def render_batch_analysis_ui(model_selector, threshold):
    st.subheader("📦 Batch Analysis")
    
    uploaded_files = st.file_uploader("Upload Multiple CT Slices", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button(f"Analyze {len(uploaded_files)} Scans"):
            model = load_model(model_selector)
            
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                try:
                    image = Image.open(file).convert("RGB")
                    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        tumor_prob = probs[0][1].item()
                    
                    results.append({
                        "Filename": file.name,
                        "Prediction": "Tumor" if tumor_prob > threshold else "Normal",
                        "Confidence": tumor_prob,
                        "Status": "⚠️ High Risk" if tumor_prob > threshold else "✅ Normal"
                    })
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
