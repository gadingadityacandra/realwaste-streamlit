
import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# 1. Judul Aplikasi Web
st.title("Pengelompokkan Jenis Makanan")
st.divider()

st.header("Area untuk Mengunggah Gambar")

# 2. Tombol Unggah Gambar (Mendukung JPG, JPEG, PNG)
uploaded_images = st.file_uploader(
    "Upload satu atau beberapa gambar sekaligus...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True # kode yang ditambah 
)
# 3. Daftar Nama Kelas Makanan 
class_names = [
    'chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 
    'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi'
]

# 4. Fungsi untuk Memuat Model TFLite agar Lebih Cepat (Caching)
@st.cache_resource
def load_model():
    # Menggunakan model milik Anda
    interpreter = Interpreter(model_path="kangadit.tflite")
    return interpreter

# Jalankan fungsi memuat model
interpreter = load_model()
interpreter.allocate_tensors()

# Mengambil informasi struktur input dan output model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 5. Proses Prediksi Jika Gambar Sudah Diunggah
if uploaded_images: # Mengecek jika list gambar tidak kosong

    st.divider()
    st.header("Area Menampilkan Hasil")
    
    # MEMBUAT KOLOM BERDASARKAN JUMLAH GAMBAR YANG DIUNGGAH
    cols = st.columns(len(uploaded_images)) # kode penambahan
    
    # PERULANGAN UNTUK MEMPROSES GAMBAR SATU PER SATU
    for index, uploaded_image in enumerate(uploaded_images): # kode perubahan
        
        # Membuka gambar satu per satu dan memastikan formatnya RGB
        image = Image.open(uploaded_image).convert("RGB")
        
        # Mengubah ukuran gambar sesuai input model (224x224)
        resized_img = image.resize((224, 224))
        
        # Mengonversi gambar ke dalam bentuk array tipe data float32
        img_array = np.array(resized_img, dtype=np.float32)
        
        # Menambah dimensi batch supaya array menjadi format [1, 224, 224, 3]
        input_data = np.expand_dims(img_array, axis=0)
        
        # Memasukkan data gambar ke model dan menjalankan prediksi
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Mengambil hasil akhir skor prediksi
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Mencari nama makanan berdasarkan skor tertinggi
        pred_class = class_names[output_data.argmax()]
        
        # 6. Menampilkan Hasil ke Dalam Kolom Masing-Masing
        with cols[index]: # kode penambahan
            st.subheader(f"Gambar {index + 1}:")
            st.write(f"**Prediksi: {pred_class}**")
            st.image(image, use_container_width=True)