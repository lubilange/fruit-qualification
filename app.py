import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIGURATION
# =========================

IMG_SIZE = 224
CLASS_NAMES = ["Bad", "Good", "Mixed"]

MODEL_PATH = "efficientnetb0_fruit_quality.keras"
FILE_ID = "1f_DZ2Iknwa2vXTTC5TkDzIXHmRPaWz3a"

st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="wide"
)

# =========================
# STYLE CSS
# =========================

# =========================
# STYLE CSS (LIGHT + DARK MODE COMPATIBLE)
# =========================

st.markdown("""
<style>

.stApp {
    background: var(--background-color);
}

/* HERO */
.hero {
    background: linear-gradient(
        120deg,
        #1b5e20,
        #43a047,
        #f9a825
    );
    padding: 34px;
    border-radius: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 14px 35px rgba(0,0,0,0.18);
    margin-bottom: 26px;
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 20px;
    margin-top: 0;
}


/* CARTES */
.card {
    background: var(--secondary-background-color);
    padding: 28px;
    border-radius: 26px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 25px;
    color: var(--text-color);
}


.small-card {
    background: var(--secondary-background-color);
    padding: 20px;
    border-radius: 22px;
    box-shadow: 0 7px 20px rgba(0,0,0,0.10);
    border-left: 7px solid #f9a825;
    margin-bottom: 18px;
    color: var(--text-color);
}


/* RESULTATS */
.result-good {
    background: rgba(76,175,80,0.18);
    color: #4caf50;
    padding:25px;
    border-radius:24px;
    text-align:center;
    font-size:28px;
    font-weight:800;
}


.result-bad {
    background: rgba(244,67,54,0.18);
    color:#ef5350;
    padding:25px;
    border-radius:24px;
    text-align:center;
    font-size:28px;
    font-weight:800;
}


.result-mixed {
    background: rgba(255,193,7,0.18);
    color:#ffc107;
    padding:25px;
    border-radius:24px;
    text-align:center;
    font-size:28px;
    font-weight:800;
}


/* FOOTER */
.footer {
    text-align:center;
    color:var(--text-color);
    opacity:0.7;
    margin-top:35px;
    font-size:14px;
}


/* JAUGE */
.gauge-center {
    background:var(--background-color);
    color:var(--text-color);
}

</style>
""", unsafe_allow_html=True)

# =========================
# CHARGEMENT DU MODÈLE
# =========================

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
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: conic-gradient(#2e7d32 {value}%, #eeeeee 0%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    ">
        <div style="
            width: 142px;
            height: 142px;
            border-radius: 50%;
            background: var(--background-color);
            color: var(--text-color);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            font-weight: 800;
        ">
            <div style="font-size: 36px;">{value}%</div>
            <div style="font-size: 14px;">Confiance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def predict_image(image, model):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(preds[predicted_idx]) * 100

    return preds, predicted_class, confidence


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🍎 Fruit Quality AI")
page = st.sidebar.radio(
    "Menu",
    ["Analyse", "À propos et limites"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Modèle")
st.sidebar.write("EfficientNetB0")
st.sidebar.write("Transfer Learning")

st.sidebar.markdown("### Performances")
st.sidebar.write("Accuracy : **94,20 %**")
st.sidebar.write("F1-score macro : **88,85 %**")

# =========================
# CHARGER MODÈLE
# =========================

model = load_model()

# =========================
# PAGE ANALYSE
# =========================

if page == "Analyse":

    st.markdown("""
    <div class="hero">
        <h1> Fruit Quality Inspection</h1>
        <p>Analyse automatique de la qualité des fruits par Intelligence Artificielle</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Objectif</h3>
        <p>
        Importez une image ou prenez une photo avec la caméra.
        Le système prédit automatiquement la qualité du fruit :
        <b>Good</b>, <b>Bad</b> ou <b>Mixed</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼️ Image à analyser")

        input_mode = st.radio(
            "Choisir la source de l'image",
            ["Importer une image", "Utiliser la caméra"],
            horizontal=True
        )

        uploaded_file = None

        if input_mode == "Importer une image":
            uploaded_file = st.file_uploader(
                "Importer une image de fruit",
                type=["jpg", "jpeg", "png", "webp"]
            )
        else:
            uploaded_file = st.camera_input("Prendre une photo avec la caméra")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image sélectionnée", use_container_width=True)
        else:
            st.info("Ajoutez une image ou prenez une photo pour lancer l'analyse.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Résultat")

        if uploaded_file is not None:
            preds, predicted_class, confidence = predict_image(image, model)

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
        st.subheader("📊 Niveau de confiance par classe")

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

# =========================
# PAGE À PROPOS
# =========================

else:

    st.markdown("""
    <div class="hero">
        <h1>ℹ️ À propos du prototype</h1>
        <p>Informations sur le modèle, les fruits supportés et les limites du système</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="small-card">
            <h3>🍎 Fruits supportés</h3>
            <p>Apple</p>
            <p>Banana</p>
            <p>Guava</p>
            <p>Lime</p>
            <p>Orange</p>
            <p>Pomegranate</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="small-card">
            <h3>🎯 Classes prédites</h3>
            <p>🟢 <b>Good</b> : bon état</p>
            <p>🔴 <b>Bad</b> : mauvais état</p>
            <p>🟡 <b>Mixed</b> : bon et mauvais état </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="small-card">
            <h3>🤖 Modèle utilisé</h3>
            <p>Architecture : <b>EfficientNetB0</b></p>
            <p>Méthode : <b>Transfer Learning</b></p>
            <p>Entrée : images 224 × 224 pixels</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>⚠️ Limites du système</h3>
        <p>
        Ce prototype a été entraîné uniquement sur six espèces de fruits :
        Apple, Banana, Guava, Lime, Orange et Pomegranate.
        Il ne doit donc pas être utilisé pour évaluer d'autres fruits ou des objets
        comme du papier, un téléphone, une bouteille ou une voiture.
        </p>

        <p>
        Le modèle ne vérifie pas automatiquement si l'image contient réellement un fruit.
        Il attribue toujours l'une des trois classes : Good, Bad ou Mixed.
        Pour une application industrielle, il serait nécessaire d'ajouter un module
        de détection préalable de fruit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📌 Recommandations d'utilisation</h3>
        <p>Utiliser une image claire contenant un seul fruit.</p>
        <p>Éviter les arrière-plans trop complexes.</p>
        <p>Utiliser uniquement les fruits supportés par le modèle.</p>
        <p>Ne pas interpréter les résultats comme une décision sanitaire officielle.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Prototype basé sur EfficientNetB0 — Contrôle intelligent de la qualité des fruits
</div>
""", unsafe_allow_html=True)
