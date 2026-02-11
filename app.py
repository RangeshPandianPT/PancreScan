import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_PATH_DENSE = "densenet121_best.pt" # Placeholder
MODEL_PATH_EFFICIENT = "outputs/demo_models/efficientnet_v2_s_fold_1_best.pt" # The one we are training
MODEL_PATH_CONVNEXT = "convnext_tiny_best.pt" # Placeholder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- App Layout ---
st.set_page_config(page_title="PancreScan AI Demo", layout="wide")
st.title("🔬 PancreScan: AI-Powered Pancreatic Cancer Detection")
st.markdown("Upload a CT scan slice to detect potential tumors using Deep Learning.")

st.sidebar.header("Settings")
model_selector = st.sidebar.selectbox(
    "Select Model Architecture",
    ["EfficientNet-V2-S (Recommended)", "DenseNet121", "ConvNeXt-Tiny"]
)
threshold_slider = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

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
        if "DenseNet121" in model_name:
            model = models.densenet121(weights=None)
            trained_model_path = MODEL_PATH_DENSE
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2), # As per training script
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
            model.classifier[2] = nn.Sequential( # Matches training code??
                # Wait, training code uses model.classifier[2] = nn.Sequential(...) OR just nn.Linear?
                # run_kfold_cv.py uses:
                # model.classifier[2] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 2)
            )
        
        if os.path.exists(trained_model_path):
            state_dict = torch.load(trained_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            st.sidebar.success(f"Loaded weights from {trained_model_path}")
        else:
            st.sidebar.warning(f"Weights file not found: {trained_model_path}. Using random weights.")
        
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Grad-CAM Logic ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook) # Note: full_backward_hook preferred in newer PyTorch

    def generate_heatmap(self, input_image, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        score = output[:, class_idx]
        score.backward()
        
        gradients = self.gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

# Note: Custom GradCAM implementation is tricky with hooks. 
# Better to use `pytorch_grad_cam` library if installed, but custom is dependency-free.
# Wait, let's look at `torch-cam` or simpler approach.
# Actually, for the hackathon, a simple occlusion map or just the prediction is fine?
# No, user explicitly approved Grad-CAM.
# Let's use `pytorch_grad_cam`?
# I installed `grad-cam`. That IS `pytorch-grad-cam`.
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer(model, model_name):
    if "DenseNet" in model_name:
        return [model.features[-1]]
    elif "EfficientNet" in model_name:
        # EfficientNet features are in model.features
        # Usually the last block
        return [model.features[-1]]
    elif "ConvNeXt" in model_name:
        # ConvNeXt features: model.features[-1]
        return [model.features[-1]]
    return None

# --- Main App Logic ---

model = load_model(model_selector)

uploaded_file = st.file_uploader("Choose a CT Scan Image", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        if model:
            # Preprocess
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
                tumor_prob = probs[0][1].item()
            
            # Display Result
            with col2:
                st.subheader("Analysis Results")
                
                if tumor_prob > threshold_slider:
                    st.error(f"**Prediction:** PANCREATIC TUMOR DETECTED")
                    st.write(f"**Confidence:** {tumor_prob*100:.2f}%")
                else:
                    st.success(f"**Prediction:** Normal Pancreas")
                    st.write(f"**Confidence (Normal):** {(1-tumor_prob)*100:.2f}%")
                
                st.progress(tumor_prob)
                
                # Grad-CAM Visualization
                try:
                    target_layers = get_target_layer(model, model_selector)
                    if target_layers:
                        cam = GradCAM(model=model, target_layers=target_layers) # Removed use_cuda=True/False, handled by device
                        # targets = [ClassifierOutputTarget(1)] # Focus on Tumor class
                        # Or focus on predicted class?
                        # Let's focus on Tumor class (1) to see what looks like a tumor.
                        targets = [ClassifierOutputTarget(1)]

                        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                        grayscale_cam = grayscale_cam[0, :]
                        
                        # Resize image to numpy float32 for visualization
                        img_np = np.array(image.resize((224, 224))) / 255.0
                        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                        
                        st.image(visualization, caption="Grad-CAM Heatmap (Tumor Focus)", use_column_width=True)
                    else:
                        st.info("Grad-CAM layer not configured for this model.")
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM: {e}")
