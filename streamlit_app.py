#Kidney Stone Detection - Streamlit App

import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
from datetime import datetime
from model import ResUNet

# PAGE CONFIG
st.set_page_config(
    page_title="Kidney Stone Detection",
    page_icon="🪨",
    layout="centered"
)
# LOAD MODEL (cached)
@st.cache_resource
def load_model():
    model_path = "models/final_model.pth"
    metadata_path = "models/model_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model = ResUNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, metadata

# PREDICTOR
class Predictor:
    def __init__(self, model, metadata):
        self.model = model
        self.threshold = metadata.get('optimal_threshold', 0.5)
        self.mean = 0.485
        self.std = 0.229
        self.input_size = 256
    
    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image / 255.0
        image = (image - self.mean) / self.std
        
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0).unsqueeze(0)
        return image
    
    def predict(self, image):
        image_tensor = self.preprocess(image)
        with torch.no_grad():
            pred = self.model(image_tensor)
        
        pred_numpy = pred.squeeze().numpy()
        binary_mask = (pred_numpy > self.threshold).astype(np.uint8)
        return binary_mask, pred_numpy
    
    def calculate_volume(self, binary_mask):
        pixel_count = binary_mask.sum()
        pixel_area_mm2 = 0.5 ** 2  # 0.5mm pixel spacing
        slice_thickness_mm = 1.0
        volume_mm3 = pixel_count * pixel_area_mm2 * slice_thickness_mm
        volume_ml = volume_mm3 / 1000
        return volume_mm3, volume_ml
    
    def create_overlay(self, original_image, binary_mask):
        if len(original_image.shape) == 2:
            overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = original_image.copy()
        
        overlay = cv2.resize(overlay, (self.input_size, self.input_size))
        overlay_copy = overlay.copy()
        overlay_copy[binary_mask > 0] = [255, 0, 0]
        blended = cv2.addWeighted(overlay, 0.6, overlay_copy, 0.4, 0)
        return blended

# LOAD MODEL
try:
    model, metadata = load_model()
    predictor = Predictor(model, metadata)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Failed to load model: {str(e)}")

# UI
st.title("🪨 Kidney Stone Detection")
st.markdown("""
**Upload a CT scan** to detect kidney stones and measure volume.
""")
# Supported formats
st.caption("📁 Supported formats: PNG, JPG, JPEG, TIFF, BMP")
# File upload
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Display uploaded image
    st.image(image, caption="Uploaded CT Scan", use_container_width=True)
    # Analyze button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not model_loaded:
            st.error("❌ Model not loaded. Please check model files.")
        else:
            with st.spinner("Analyzing..."):
                # Predict
                binary_mask, _ = predictor.predict(image_np)
                # Calculate volume
                volume_mm3, volume_ml = predictor.calculate_volume(binary_mask)
                # Create overlay
                overlay = predictor.create_overlay(image_np, binary_mask)
                
                # DISPLAY RESULTS
                # Volume metrics
                col1, col2 = st.columns(2)
                col1.metric("Volume", f"{volume_mm3:.1f}", "mm³")
                col2.metric("Volume", f"{volume_ml:.3f}", "mL")
                
                # Images
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # ✅ FIX: Scale mask to 0-255 for display
                    mask_display = (binary_mask * 255).astype(np.uint8)
                    st.image(mask_display, caption="Mask", use_container_width=True, clamp=True)
                
                with col2:
                    st.image(overlay, caption="Overlay", use_container_width=True, clamp=True)
                
                # Download buttons
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
                    buf = BytesIO()
                    mask_pil.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Mask",
                        data=buf.getvalue(),
                        file_name=f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col2:
                    overlay_pil = Image.fromarray(overlay)
                    buf = BytesIO()
                    overlay_pil.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Overlay",
                        data=buf.getvalue(),
                        file_name=f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
else:
    # Placeholder when no image uploaded
    st.markdown("""
    <div style="
        border: 2px dashed #d1d5db;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        color: #9ca3af;
    ">
        <p style="font-size: 3rem;">🪨</p>
        <p>Upload a CT scan to begin</p>
    </div>
    """, unsafe_allow_html=True)
# FOOTER
st.markdown("---")
st.caption("⚠️ For clinical use with radiologist supervision")