import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ["Bad", "Good", "Mixed"]

MODEL_PATH = "efficientnetb0_fruit_quality.keras"
FILE_ID = "1f_DZ2Iknwa2vXTTC5TkDzIXHmRPaWz3a"

st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff7e6 0%, #ffe4b5 35%, #e8ffd8 100%);
}

.hero {
    background: linear-gradient(120deg, #2e7d32, #66bb6a, #ffb300);
    padding: 35px;
    border-radius: 28px;
    color: white;
    text-align: center;
    box-shadow: 0 12px 35px rgba(0,0,0,0.18);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 46px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 20px;
    margin-top: 0;
}

.card {
    background: white;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    border: 1px solid #ffe0a3;
    margin-bottom: 25px;
}

.result-good {
    background: linear-gradient(135deg, #d8f5d0, #b9f6ca);
    color: #1b5e20;
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}

.result-bad {
    background: linear-gradient(135deg, #ffd6d6, #ffb3b3);
    color: #b71c1c;
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}

.result-mixed {
    background: linear-gradient(135deg, #fff3cd, #ffe082);
    color: #8a6d00;
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}

.footer {
    text-align: center;
    color: #555;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle IA..."):
            gdown.download(
                id=FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    return tf.keras.models.load_model(MODEL_PATH)


def circular_gauge(confidence):
    value = round(confidence, 2)

    st.markdown(f"""
    <div style="
        width: 190px;
        height: 190px;
        border-radius: 50%;
        background: conic-gradient(#2e7d32 {value}%, #eeeeee 0%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    ">
        <div style="
            width: 135px;
            height: 135px;
            border-radius: 50%;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            font-weight: 800;
            color: #2e7d32;
        ">
            <div style="font-size: 34px;">{value}%</div>
            <div style="font-size: 14px;">Confiance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


model = load_model()

st.markdown("""
<div class="hero">
    <h1>Fruit Quality Inspection</h1>
    <p>Analyse automatique de la qualité des fruits par Intelligence Artificielle</p>
</div>
""", unsafe_allow_html=True)

st.info("""
🍎 **Fruits supportés**
Pomme • Banane • Goyave • Citron vert • Orange • Grenade

🎯 **Le modèle prédit uniquement la qualité**

🟢 **Good** : Bon état  
🔴 **Bad** : Mauvais état  
🟡 **Mixed** : Bon et mauvais état

⚠️ **Limitation**

Le modèle fonctionne uniquement avec ces six espèces de fruits.  
Les objets comme le papier, le téléphone, la bouteille ou la voiture peuvent produire des prédictions non fiables.
""")

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Image à analyser")

    uploaded_file = st.file_uploader(
        "Importer une image de fruit",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image importée", use_container_width=True)
    else:
        st.info("Ajoute une image pour lancer l'analyse.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Résultat de prédiction")

    if uploaded_file is not None:
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(preds[predicted_idx]) * 100

        if predicted_class == "Good":
            st.markdown(
                f'<div class="result-good">🟢 Fruit de bonne qualité<br>{confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        elif predicted_class == "Bad":
            st.markdown(
                f'<div class="result-bad">🔴 Fruit de mauvaise qualité<br>{confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-mixed">🟡 Qualité mixte<br>{confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

        st.write("")
        circular_gauge(confidence)

    else:
        st.warning("Aucune image analysée pour le moment.")

    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Niveau de confiance par classe")

    for classe, prob in zip(CLASS_NAMES, preds):
        if classe == "Good":
            emoji = "🟢"
            label = "Good - Bon état"
        elif classe == "Bad":
            emoji = "🔴"
            label = "Bad - Mauvais état"
        else:
            emoji = "🟡"
            label = "Mixed - Bon et mauvais état"

        st.write(f"{emoji} **{label}** : {prob * 100:.2f}%")
        st.progress(float(prob))

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Prototype basé sur EfficientNetB0 — Contrôle intelligent de la qualité des fruits
</div>
""", unsafe_allow_html=True)
