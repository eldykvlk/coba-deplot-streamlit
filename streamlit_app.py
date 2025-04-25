import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Nama aplikasi
st.set_page_config(page_title="ReFisher AI", layout="centered")
st.title("🐟 ReFisher AI")
st.subheader("Klasifikasi Ikan Segar dan Tidak Segar")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_ikan.h5")
    return model

model = load_model()

# Ukuran input model
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Fungsi prediksi
def predict_image(img):
    st.write("🔧 Mengubah ukuran gambar...")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    st.write("🔧 Konversi ke array...")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    st.write("🔧 Dimensi array gambar:", img_array.shape)
    img_array = tf.expand_dims(img_array, 0)
    st.write("🔧 Dimensi setelah expand_dims:", img_array.shape)
    st.write("🔧 Preprocessing...")
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    st.write("🚀 Memanggil model.predict()...")
    predictions = model.predict(img_array)
    st.write("🔧 Prediksi:", predictions)

    score = predictions[0][0]
    label = "SEGAR" if score > 0.5 else "TIDAK SEGAR"
    confidence = score if score > 0.5 else 1 - score
    st.write(f"✅ Prediksi selesai: {label} ({confidence:.2f})")

    return label, confidence

# UI untuk unggah atau ambil gambar
option = st.radio("Pilih metode input gambar:", ("📤 Upload Gambar", "📸 Ambil Gambar (Kamera)"))

if option == "📤 Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar ikan...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.write("✅ Gambar berhasil diunggah")
        img = Image.open(uploaded_file)
        img = img.convert("RGB")
        st.image(img, caption='Gambar yang diunggah', use_container_width=True)

        st.write("⏳ Menjalankan prediksi...")
        try:
            label, confidence = predict_image(img)
            st.markdown(f"### Prediksi: **{label}**")
            st.metric(label="Akurasi Prediksi", value=f"{confidence*100:.2f}%")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

elif option == "📸 Ambil Gambar (Kamera)":
    camera_img = st.camera_input("Ambil gambar dengan kamera")
    if camera_img is not None:
        img = Image.open(camera_img)
        st.image(img, caption="Gambar dari kamera", use_container_width=True)

        st.write("⏳ Menjalankan prediksi...")
        try:
            label, confidence = predict_image(img)
            st.markdown(f"### Prediksi: **{label}**")
            st.metric(label="Akurasi Prediksi", value=f"{confidence*100:.2f}%")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# Footer
st.caption("© 2025 ReFisher AI 🧠🐟")
