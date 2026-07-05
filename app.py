import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

IMG_SIZE = 224
CLASS_NAMES = ["Bad", "Good", "Mixed"]

st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff7e6 0%, #ffe8cc 40%, #e8ffd8 100%);
}
.main-card {
    background: white;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    border: 2px solid #ffb347;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #2e7d32;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
}
.result-good {
    background: #d8f5d0;
    color: #1b5e20;
    padding: 18px;
    border-radius: 16px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
.result-bad {
    background: #ffd6d6;
    color: #b71c1c;
    padding: 18px;
    border-radius: 16px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
.result-mixed {
    background: #fff3cd;
    color: #8a6d00;
    padding: 18px;
    border-radius: 16px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetb0_fruit_quality.keras")

model = load_model()

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">🍓 Fruit Quality AI 🍊</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Système intelligent de contrôle de la qualité des fruits</div>',
    unsafe_allow_html=True
)

st.write("")
uploaded_file = st.file_uploader(
    "📤 Importer une image de fruit",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Image importée", use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = preds[predicted_idx] * 100

    if predicted_class == "Good":
        st.markdown(
            f'<div class="result-good">✅ Fruit de bonne qualité<br>{confidence:.2f}% de confiance</div>',
            unsafe_allow_html=True
        )
    elif predicted_class == "Bad":
        st.markdown(
            f'<div class="result-bad">❌ Fruit de mauvaise qualité<br>{confidence:.2f}% de confiance</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-mixed">⚠️ Qualité mixte<br>{confidence:.2f}% de confiance</div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.subheader("📊 Probabilités par classe")

    prob_df = pd.DataFrame({
        "Classe": CLASS_NAMES,
        "Probabilité (%)": [round(float(p) * 100, 2) for p in preds]
    })

    st.bar_chart(prob_df.set_index("Classe"))

    with st.expander("Voir les détails numériques"):
        st.dataframe(prob_df, use_container_width=True)

else:
    st.info("Ajoute une image pour lancer la prédiction.")

st.markdown("</div>", unsafe_allow_html=True)

st.caption("Prototype basé sur EfficientNetB0 — Contrôle intelligent de la qualité des fruits.")
