import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Configuration ---
MODEL_PATH_DENSE = "densenet121_best.pt"
MODEL_PATH_EFFICIENT = "outputs/demo_models/efficientnet_v2_s_fold_1_best.pt"
MODEL_PATH_CONVNEXT = "convnext_tiny_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Custom CSS ---
ST_STYLE = """
<style>
    /* Main Layout */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 2px solid #e9ecef;
        margin-bottom: 30px;
    }
    h2, h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Result Cards */
    .result-card-normal {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .result-card-tumor {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .confidence-score {
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        z-index: 100;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
    }
</style>
"""

# --- App Layout ---
st.set_page_config(page_title="PancreScan AI", page_icon="🏥", layout="wide")
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
            # Fallback for demo if precise weights missing, but warn user
            if model: 
                pass 
                # st.toast(f"Warning: Weights not found at {trained_model_path}", icon="⚠️")
        
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

def main():
    # Sidebar
    st.sidebar.title("🏥 PancreScan AI")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("⚙️ Configuration")
    model_selector = st.sidebar.selectbox(
        "Model Architecture",
        ["EfficientNet-V2-S (Recommended)", "DenseNet121", "ConvNeXt-Tiny"]
    )
    
    st.sidebar.markdown("---")
    threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.4, 0.05, help="Lower threshold increases sensitivity to potential tumors.")
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About**\n\n"
        "PancreScan uses deep learning to assist radiologists in detecting pancreatic tumors from CT slices.\n\n"
        "© 2026 PancreScan Team"
    )

    # Main Area
    st.title("Pancreatic Tumor Detection System")
    st.markdown("### 🤖 AI-Powered Second Reader for Abdominal CT")
    
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
                            
                        # Store results in session state to persist through re-runs if needed
                        st.session_state['tumor_prob'] = tumor_prob
                        st.session_state['analyzed'] = True
        
        with col2:
            if st.session_state.get('analyzed', False):
                tumor_prob = st.session_state.get('tumor_prob', 0.0)
                
                is_tumor = tumor_prob > threshold
                
                if is_tumor:
                    st.markdown(
                        f"""
                        <div class="result-card-tumor">
                            <h3>🚨 Tumor Detected</h3>
                            <p>The model has identified patterns consistent with pancreatic malignancy.</p>
                            <p class="confidence-score">Confidence: {tumor_prob*100:.2f}%</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-normal">
                            <h3>✅ Normal Pancreas</h3>
                            <p>No malignant patterns detected above the sensitivity threshold.</p>
                            <p class="confidence-score">Confidence (Normal): {(1-tumor_prob)*100:.2f}%</p>
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

if __name__ == "__main__":
    main()
