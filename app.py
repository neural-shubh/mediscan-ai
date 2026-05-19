import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from huggingface_hub import hf_hub_download
from PIL import Image
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

* { font-family: 'Syne', sans-serif; }

.stApp { background: #080c10; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2d3d;
}

.main-header { text-align: center; padding: 2rem 0 1rem 0; }
.main-header h1 {
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #00ff88 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px; margin: 0;
}
.main-header p {
    color: #64748b; font-size: 1rem; margin-top: 0.5rem;
    letter-spacing: 2px; text-transform: uppercase;
}

.result-card { background: #0d1117; border: 1px solid #1e2d3d; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
.result-card.success { border-color: #00ff88; box-shadow: 0 0 20px rgba(0,255,136,0.1); }
.result-card.warning { border-color: #ffd700; box-shadow: 0 0 20px rgba(255,215,0,0.1); }
.result-card.danger  { border-color: #ff4757; box-shadow: 0 0 20px rgba(255,71,87,0.1); }

.pred-label { font-size: 2rem; font-weight: 800; color: #00d4ff; margin: 0; }
.confidence-high { color: #00ff88; }
.confidence-mid  { color: #ffd700; }
.confidence-low  { color: #ff4757; }

.prob-bar-container { margin: 0.4rem 0; }
.prob-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 2px; font-family: 'DM Mono', monospace; }
.prob-bar-bg { background: #1e2d3d; border-radius: 4px; height: 8px; width: 100%; }
.prob-bar-fill { height: 8px; border-radius: 4px; }

.warning-box { background: rgba(255,215,0,0.05); border: 1px solid #ffd700; border-radius: 8px; padding: 1rem; color: #ffd700; font-size: 0.9rem; margin-top: 1rem; }

.badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.badge-brain { background: rgba(0,212,255,0.1); color: #00d4ff; border: 1px solid #00d4ff; }
.badge-chest { background: rgba(0,255,136,0.1); color: #00ff88; border: 1px solid #00ff88; }

.divider { border: none; border-top: 1px solid #1e2d3d; margin: 1.5rem 0; }
.upload-hint { color: #475569; font-size: 0.85rem; text-align: center; margin-top: 0.5rem; font-family: 'DM Mono', monospace; }
.metric-box { background: #0d1117; border: 1px solid #1e2d3d; border-radius: 8px; padding: 1rem; text-align: center; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #00d4ff; }
.metric-label { font-size: 0.75rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOADING FROM HUGGING FACE
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models..."):
        brain_path = hf_hub_download(
            repo_id="neural-shubh/mediscan-models",
            filename="brain_model_final.keras"
        )
        chest_path = hf_hub_download(
            repo_id="neural-shubh/mediscan-models",
            filename="chest_model_final.keras"
        )
        return {
            'brain': load_model(brain_path),
            'chest': load_model(chest_path)
        }


# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    try:
        base_model = model.layers[0]
        dummy = tf.zeros((1, *img_array.shape[1:]))
        base_model(dummy)
        model(dummy)

        extractor = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[
                base_model.get_layer(last_conv_layer_name).output,
                base_model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_out, base_out = extractor(img_array)
            tape.watch(conv_out)
            x = base_out
            for layer in model.layers[1:]:
                x = layer(x)
            predicted_class = tf.argmax(x[0])
            class_score = x[:, predicted_class]

        grads = tape.gradient(class_score, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

        x = base_model(img_array)
        for layer in model.layers[1:]:
            x = layer(x)

        return heatmap.numpy(), np.array(x)
    except Exception:
        predictions = model.predict(img_array, verbose=0)
        return None, predictions


def overlay_gradcam(img_array_orig, heatmap, target_size):
    img = cv2.resize(img_array_orig, target_size)
    heatmap_resized = cv2.resize(heatmap, target_size)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img.astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)
    return superimposed


# ─────────────────────────────────────────────
#  MODEL CONFIG
# ─────────────────────────────────────────────
MODEL_CONFIG = {
    'brain': {
        'class_labels'   : ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'preprocess_fn'  : efficientnet_v2.preprocess_input,
        'target_size'    : (300, 300),
        'last_conv_layer': 'top_activation',
        'binary'         : False,
    },
    'chest': {
        'class_labels'   : ['Normal', 'Pneumonia'],
        'preprocess_fn'  : densenet_preprocess,
        'target_size'    : (224, 224),
        'last_conv_layer': 'relu',
        'binary'         : True,
    }
}


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
def run_prediction(uploaded_file, modality, model):
    config      = MODEL_CONFIG[modality]
    target_size = config['target_size']
    labels      = config['class_labels']
    binary      = config['binary']

    pil_img  = Image.open(uploaded_file).convert('RGB').resize(target_size)
    img_orig = np.array(pil_img)

    img_array = config['preprocess_fn'](img_orig.astype(np.float32).copy())
    img_array = np.expand_dims(img_array, axis=0)

    heatmap, predictions = get_gradcam_heatmap(model, img_array, config['last_conv_layer'])

    if binary:
        prob_pneumonia = float(predictions[0][0])
        prob_normal    = 1 - prob_pneumonia
        probs          = [prob_normal, prob_pneumonia]
        predicted_idx  = 1 if prob_pneumonia > 0.5 else 0
        confidence     = max(probs) * 100
    else:
        probs         = list(predictions[0])
        predicted_idx = int(np.argmax(probs))
        confidence    = float(np.max(probs)) * 100

    gradcam_img = overlay_gradcam(img_orig, heatmap, target_size) if heatmap is not None else None

    return {
        'predicted_class': labels[predicted_idx],
        'predicted_idx'  : predicted_idx,
        'confidence'     : confidence,
        'probs'          : probs,
        'labels'         : labels,
        'original_img'   : img_orig,
        'gradcam_img'    : gradcam_img,
    }


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0'>
        <div style='font-size:1.4rem;font-weight:800;color:#00d4ff'>MediScan AI</div>
        <div style='font-size:0.75rem;color:#475569;letter-spacing:2px;text-transform:uppercase'>Medical Image Classifier</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("**Select Scan Type**")
    modality     = st.radio("", options=["Brain MRI", "Chest X-Ray"], label_visibility="collapsed")
    modality_key = 'brain' if modality == "Brain MRI" else 'chest'

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Classes**")
    for label in MODEL_CONFIG[modality_key]['class_labels']:
        st.markdown(f"<div style='font-size:0.85rem;color:#94a3b8;padding:2px 0'>→ {label}</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#334155'>
        ⚠️ For research purposes only.<br>Not a substitute for professional medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>MediScan AI</h1>
    <p>Neural Medical Image Analysis · Grad-CAM Explainability</p>
</div>
""", unsafe_allow_html=True)

badge_class = "badge-brain" if modality_key == 'brain' else "badge-chest"
st.markdown(f"<div style='text-align:center;margin-bottom:2rem'><span class='badge {badge_class}'>{modality}</span></div>",
            unsafe_allow_html=True)

models = load_models()

uploaded_file = st.file_uploader(f"Upload {modality} image", type=["jpg","jpeg","png"], label_visibility="collapsed")
st.markdown(f"<div class='upload-hint'>Supported: JPG, JPEG, PNG · {modality}</div>", unsafe_allow_html=True)

if uploaded_file:
    with st.spinner("Analysing image..."):
        result = run_prediction(uploaded_file, modality_key, models[modality_key])

    confidence = result['confidence']
    predicted  = result['predicted_class']

    if confidence >= 85:
        conf_class, card_class = "confidence-high", "success"
    elif confidence >= 70:
        conf_class, card_class = "confidence-mid", "warning"
    else:
        conf_class, card_class = "confidence-low", "danger"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        st.image(result['original_img'], use_column_width=True)

    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        if result['gradcam_img'] is not None:
            st.image(result['gradcam_img'], use_column_width=True)
            st.markdown("<div style='font-size:0.75rem;color:#475569;text-align:center'>🔴 Red = High attention &nbsp;|&nbsp; 🔵 Blue = Low attention</div>",
                        unsafe_allow_html=True)
        else:
            st.info("Grad-CAM unavailable — showing prediction only")

    with col3:
        st.markdown("**Analysis Result**")
        st.markdown(f"""
        <div class='result-card {card_class}'>
            <div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem'>Prediction</div>
            <div class='pred-label'>{predicted}</div>
            <div style='margin-top:0.8rem'>
                <span style='font-size:0.75rem;color:#475569'>Confidence</span><br>
                <span class='{conf_class}' style='font-size:1.5rem;font-weight:700'>{confidence:.1f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Class Probabilities**")
        for label, prob in zip(result['labels'], result['probs']):
            pct     = float(prob) * 100
            is_pred = label == predicted
            st.markdown(f"""
            <div class='prob-bar-container'>
                <div class='prob-label'>{label} — {pct:.1f}%</div>
                <div class='prob-bar-bg'>
                    <div class='prob-bar-fill' style='width:{pct}%;background:{"linear-gradient(90deg,#00d4ff,#00ff88)" if is_pred else "#1e4060"}'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        if confidence < 70:
            st.markdown("<div class='warning-box'>⚠️ <strong>Low confidence.</strong><br>Please consult a radiologist.</div>",
                        unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    gradcam_status = "Yes" if result['gradcam_img'] is not None else "No"
    metric_values = [f"{confidence:.1f}%", str(len(result['labels'])), gradcam_status, modality_key.upper()]
    metric_labels = ["Confidence", "Classes", "Grad-CAM", "Modality"]
    for col, val, lbl in zip([m1, m2, m3, m4], metric_values, metric_labels):
        with col:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{val}</div><div class='metric-label'>{lbl}</div></div>",
                        unsafe_allow_html=True)
     

for col, val, lbl in zip([m1, m2, m3, m4], metric_values, metric_labels):
        with col:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{val}</div><div class='metric-label'>{lbl}</div></div>",
                        unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#334155'>
        <div style='font-size:4rem'>🩻</div>
        <div style='font-size:1.2rem;font-weight:600;color:#475569;margin-top:1rem'>Upload a medical image to begin analysis</div>
        <div style='font-size:0.85rem;color:#334155;margin-top:0.5rem'>Select scan type from the sidebar, then upload your image</div>
    </div>
    """, unsafe_allow_html=True)
