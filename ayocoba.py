
import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# 1. Judul Aplikasi Web
st.title("Aplikasi Klasifikasi Sampah (RealWaste)")
st.divider()

st.header("Area untuk Mengunggah Gambar Sampah")

# Tombol Unggah Gambar (Mendukung banyak gambar sekaligus)
uploaded_images = st.file_uploader(
    "Upload satu atau beberapa gambar sampah...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# 2. Daftar Nama Kelas Dataset RealWaste

class_names = [
    'Cardboard', 
    'Food Waste', 
    'Glass', 
    'Metal', 
    'Miscellaneous', 
    'Paper', 
    'Plastic', 
    'Textiles', 
    'Vegetation'
]

# 3. Memuat Model TFLite RealWaste Anda
@st.cache_resource
def load_model():
    # Pastikan nama file model .tflite Anda sudah sesuai di sini
    interpreter = Interpreter(model_path="model_waste.tflite") 
    return interpreter

interpreter = load_model()
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4. Proses Prediksi Menggunakan Perulangan (Looping)
if uploaded_images:
    st.divider()
    st.header("Area Menampilkan Hasil Klasifikasi")
    
    # Membuat kolom grid berdampingan sesuai jumlah gambar
    cols = st.columns(len(uploaded_images))
    
    for index, uploaded_image in enumerate(uploaded_images):
        # Membuka gambar dan memastikan formatnya RGB
        image = Image.open(uploaded_image).convert("RGB")
        
        # Pra-pemrosesan Gambar (Resize menjadi 224x224 sesuai model transfer learning Anda)
        resized_img = image.resize((224, 224))
        img_array = np.array(resized_img, dtype=np.float32)
        input_data = np.expand_dims(img_array, axis=0)
        
        # Menjalankan Prediksi Model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Mengambil hasil skor tertinggi
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_class = class_names[output_data.argmax()]
        
        # Menampilkan ke kolom masing-masing
        with cols[index]:
            st.subheader(f"Sampah {index + 1}:")
            st.write(f"**Kategori: {pred_class}**")
            st.image(image, use_container_width=True)