"""
Streamlit Web Application for Diabetic Retinopathy Detection & Clinical Triage.
Features interactive image upload, InceptionV3 prediction, Grad-CAM visualization,
and Ben Graham contrast enhancement toggle.
"""

import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

from src.preprocessing import ben_graham_enhancement, inception_preprocess
from src.predict import SEVERITY_STAGES
from src.gradcam import make_gradcam_heatmap, overlay_gradcam

st.set_page_config(
    page_title="Diabetic Retinopathy Detection AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👁️ Diabetic Retinopathy Diagnostic Assistant")
st.markdown("""
An automated clinical screening system powered by **InceptionV3 Deep Learning** to classify 
diabetic retinopathy severity from digital fundus photography.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")
model_path = st.sidebar.text_input("Model Checkpoint Path", "models/best_model.keras")
apply_enhancement = st.sidebar.checkbox("Apply Ben Graham Contrast Filter", value=True)
show_gradcam = st.sidebar.checkbox("Generate Grad-CAM Attention Heatmap", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 ICDR Severity Scale
- **Stage 0**: No DR
- **Stage 1**: Mild Non-Proliferative
- **Stage 2**: Moderate Non-Proliferative
- **Stage 3**: Severe Non-Proliferative
- **Stage 4**: Proliferative DR
""")


@st.cache_resource
def load_cached_model(path: str):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None


model = load_cached_model(model_path)

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image (.jpg, .png, .tiff)",
    type=["png", "jpg", "jpeg", "tiff", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    raw_arr = np.array(image)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("📷 Original Retinal Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔬 Preprocessed Fundus")
        if apply_enhancement:
            enhanced_arr = ben_graham_enhancement(raw_arr, target_size=(299, 299))
            st.image(enhanced_arr, caption="Ben Graham Illumination Normalized", use_container_width=True)
        else:
            enhanced_arr = cv2.resize(raw_arr, (299, 299))
            st.image(enhanced_arr, caption="Standard Resize (299x299)", use_container_width=True)

    # Prediction Section
    if model is None:
        st.warning(
            f"Model checkpoint not found at `{model_path}`. "
            "Please train the model first using `python -m src.train`."
        )
        # Mock simulation for preview
        simulated_probs = [0.05, 0.10, 0.70, 0.12, 0.03]
        pred_class = int(np.argmax(simulated_probs))
        info = SEVERITY_STAGES[pred_class]
        confidence = simulated_probs[pred_class]
    else:
        norm_input = inception_preprocess(enhanced_arr)
        batch_input = np.expand_dims(norm_input, axis=0)
        preds = model.predict(batch_input, verbose=0)[0]
        pred_class = int(np.argmax(preds))
        confidence = float(preds[pred_class])
        info = SEVERITY_STAGES[pred_class]
        simulated_probs = preds

    with col3:
        st.subheader("🎯 Grad-CAM Attention")
        if show_gradcam and model is not None:
            try:
                norm_input = inception_preprocess(enhanced_arr)
                heatmap = make_gradcam_heatmap(np.expand_dims(norm_input, axis=0), model)
                temp_path = "/tmp/temp_retina.png"
                image.save(temp_path)
                cam_img = overlay_gradcam(temp_path, heatmap)
                st.image(cam_img, caption="Lesion Focus Areas", use_container_width=True)
            except Exception as e:
                st.info("Grad-CAM visualization available after model training.")
        else:
            st.info("Grad-CAM overlay enabled in sidebar.")

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Diagnosis Summary")
        badge_colors = {
            0: "🟢",
            1: "🟡",
            2: "🟠",
            3: "🔴",
            4: "🚨",
        }
        st.markdown(f"### {badge_colors.get(pred_class, '⚪')} **{info['name']}** (Stage {pred_class})")
        st.metric(label="Model Diagnostic Confidence", value=f"{confidence:.2%}")
        st.markdown(f"**Clinical Finding:** {info['findings']}")
        st.markdown(f"**Recommended Action:** *{info['clinical_action']}*")

    with res_col2:
        st.subheader("Severity Probability Distribution")
        for i, p in enumerate(simulated_probs):
            stage_meta = SEVERITY_STAGES[i]
            st.write(f"**{stage_meta['name']}** ({p:.1%})")
            st.progress(min(float(p), 1.0))

else:
    st.info("👆 Upload a retinal fundus photograph to begin diagnostic evaluation.")
