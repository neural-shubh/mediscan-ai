from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import tensorflow as tf
import cv2
import io
import base64
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from huggingface_hub import hf_hub_download
import os

app = FastAPI(title="MediScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ──
print("Loading models from Hugging Face...")
brain_path = hf_hub_download(repo_id="neural-shubh/mediscan-models", filename="brain_model_final.keras")
chest_path = hf_hub_download(repo_id="neural-shubh/mediscan-models", filename="chest_model_final.keras")
brain_model = load_model(brain_path)
chest_model = load_model(chest_path)
print("Models loaded ✅")

MODEL_CONFIG = {
    "brain": {
        "class_labels"   : ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "preprocess_fn"  : efficientnet_v2.preprocess_input,
        "target_size"    : (300, 300),
        "last_conv_layer": "top_activation",
        "binary"         : False,
        "model"          : None,
    },
    "chest": {
        "class_labels"   : ["Normal", "Pneumonia"],
        "preprocess_fn"  : densenet_preprocess,
        "target_size"    : (224, 224),
        "last_conv_layer": "relu",
        "binary"         : True,
        "model"          : None,
    }
}

MODEL_CONFIG["brain"]["model"] = brain_model
MODEL_CONFIG["chest"]["model"] = chest_model


def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    try:
        base_model = model.layers[0]
        dummy = tf.zeros((1, *img_array.shape[1:]))
        base_model(dummy)
        model(dummy)
        extractor = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
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


def overlay_gradcam(img_array, heatmap, target_size):
    img = cv2.resize(img_array, target_size)
    heatmap_resized = cv2.resize(heatmap, target_size)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img.astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)
    return superimposed


def img_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/predict/{modality}")
async def predict(modality: str, file: UploadFile = File(...)):
    if modality not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid modality. Use 'brain' or 'chest'")

    config      = MODEL_CONFIG[modality]
    model       = config["model"]
    target_size = config["target_size"]
    labels      = config["class_labels"]
    binary      = config["binary"]

    contents  = await file.read()
    pil_img   = Image.open(io.BytesIO(contents)).convert("RGB").resize(target_size)
    img_orig  = np.array(pil_img)

    img_array = config["preprocess_fn"](img_orig.astype(np.float32).copy())
    img_array = np.expand_dims(img_array, axis=0)

    heatmap, predictions = get_gradcam_heatmap(model, img_array, config["last_conv_layer"])

    if binary:
        prob_pneumonia = float(predictions[0][0])
        prob_normal    = 1 - prob_pneumonia
        probs          = [prob_normal, prob_pneumonia]
        predicted_idx  = 1 if prob_pneumonia > 0.5 else 0
        confidence     = max(probs) * 100
    else:
        probs         = [float(p) for p in predictions[0]]
        predicted_idx = int(np.argmax(probs))
        confidence    = float(np.max(probs)) * 100

    gradcam_b64  = None
    if heatmap is not None:
        gradcam_img = overlay_gradcam(img_orig, heatmap, target_size)
        gradcam_b64 = img_to_base64(gradcam_img)

    original_b64 = img_to_base64(img_orig)

    return {
        "predicted_class": labels[predicted_idx],
        "confidence"     : round(confidence, 2),
        "probabilities"  : {label: round(prob * 100, 2) for label, prob in zip(labels, probs)},
        "original_image" : original_b64,
        "gradcam_image"  : gradcam_b64,
        "low_confidence" : confidence < 70,
    }


# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
