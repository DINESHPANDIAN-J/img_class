import os
import cv2
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from PIL import Image
from scipy.sparse import csr_matrix

# === Classes List ===
CLASSES = [
    'A1', 'A2', 'A3', 'A3.5', 'A4', 'B1', 'B2', 'B3', 'B4',
    'C1', 'C2', 'C3', 'C4', 'D2', 'D3', 'D4'
]

# === Feature Extraction ===

def compute_color_histogram(image):
    chans = cv2.split(image)
    hist_values = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_values.extend(hist)
    return hist_values

def compute_color_moments(image):
    chans = cv2.split(image)
    moments = []
    for chan in chans:
        mean = np.mean(chan)
        std = np.std(chan)
        moments.extend([mean, std])
    return moments

def preprocess_image(image):
    if image is None:
        st.error("Image is empty")
        return None

    # ✅ Resize to 128x128 (same as original preprocessing)
    image = cv2.resize(image, (128, 128))

    hist_values = compute_color_histogram(image)
    color_moments = compute_color_moments(image)
    return hist_values + color_moments

# === Load Model and Predict ===

@st.cache_resource
def load_model_components():
    with open('trained_models\K-Nearest_Neighbors_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('trained_models/svd_transformer.pkl', 'rb') as f:
        svd = pickle.load(f)
    with open('trained_models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, svd, label_encoder

def predict(image_features):
    model, svd, encoder = load_model_components()
    X_sparse = csr_matrix(image_features)
    X_reduced = svd.transform(X_sparse)

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X_reduced)
        probs = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
    else:
        probs = model.predict_proba(X_reduced)

    return probs, encoder

# === Plotting Function ===

def plot_predictions(image, image_name, top3_classes, top3_probs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title(image_name)
    ax[0].axis('off')

    bars = ax[1].barh(top3_classes[::-1], top3_probs[::-1], color='skyblue')
    ax[1].set_xlabel('Probability')
    ax[1].set_title('Top 3 Predictions')
    for bar, prob in zip(bars, top3_probs[::-1]):
        ax[1].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob:.2%}', va='center', ha='left')

    st.pyplot(fig)

# === Main App ===

def main():
    st.set_page_config(page_title="Dental Shade Classifier", layout="centered")
    st.title("🦷 Dental Shade Classification")
    st.write("Upload a tooth image, crop it, and get shade predictions.")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        st.subheader("🖼️ Crop Area of Interest")
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color='#FF4B4B',
            aspect_ratio=(1, 1),
        )

        st.image(cropped_img, caption="Cropped Image", use_container_width=True)

        if st.button("🔍 Predict"):
            image_cv = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)

            # ✅ Preprocess with resize to 128x128
            features = preprocess_image(image_cv)

            if features:
                features_array = np.array([features])
                probs, encoder = predict(features_array)

                top3_idx = np.argsort(probs[0])[::-1][:3]
                top3_classes = encoder.inverse_transform(top3_idx)
                top3_probs = probs[0][top3_idx]

                st.subheader("📊 Top Predictions")
                for i in range(3):
                    st.write(f"**{top3_classes[i]}**: {top3_probs[i]:.2%}")

                plot_predictions(image_cv, uploaded_file.name, top3_classes, top3_probs)

if __name__ == "__main__":
    main()