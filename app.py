import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ["Bad", "Good", "Mixed"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetb0_fruit_quality.keras")

model = load_model()

st.set_page_config(page_title="Fruit Quality Control", layout="centered")

st.title("🍎 Contrôle intelligent de la qualité des fruits")
st.write("Prototype de classification : Bad, Good ou Mixed")

uploaded_file = st.file_uploader(
    "Importer une image de fruit",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image importée", use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.subheader("Résultat")
    st.write(f"**Classe prédite : {predicted_class}**")
    st.write(f"**Confiance : {confidence:.2f}%**")

    st.subheader("Probabilités")
    for cls, prob in zip(CLASS_NAMES, predictions[0]):
        st.write(f"{cls} : {prob*100:.2f}%")
        st.progress(float(prob))
