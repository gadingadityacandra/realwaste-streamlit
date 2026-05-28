import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# Mengatur konfigurasi halaman agar responsif dan modern
st.set_page_config(
    page_title="RealWaste AI Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR (SISI KIRI WEB) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2E7D32;'>🌱 Pusat Informasi</h2>", unsafe_allow_html=True)
    st.write("Aplikasi ini menggunakan teknologi **Kecerdasan Buatan (AI)** untuk mendeteksi 9 kategori jenis sampah secara instan guna mendukung gerakan *Eco-Friendly*.")
    st.divider()
    
    st.markdown("### 📋 9 Kategori Sampah yang Didukung:")
    st.info("📦 Cardboard\n\n🍎 Food Waste\n\n🍷 Glass\n\n🔩 Metal\n\n🌀 Miscellaneous\n\n📄 Paper\n\n🥤 Plastic\n\n👕 Textiles\n\n🌿 Vegetation")
    st.divider()
    # Penanda hak cipta di sidebar kiri
    st.markdown("<center style='color: #757575; font-size: 14px;'>🛠️ Developed by <b>Kang Adit</b></center>", unsafe_allow_html=True)
    st.caption("<center>Tugas Praktikum Machine Learning v2.0</center>", unsafe_allow_html=True)

# --- HALAMAN UTAMA ---
# Desain Header Atas menggunakan HTML dengan penanda "By Kang Adit"
st.markdown("""
    <div style="background-color:#E8F5E9; padding:20px; border-radius:15px; margin-bottom:25px; border-left: 8px solid #2E7D32;">
        <h1 style="color:#1B5E20; margin:0;">♻️ Sistem Klasifikasi Sampah Otomatis</h1>
        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
            <p style="color:#4E342E; font-size:16px; margin:0;">Dataset: RealWaste | Model: TFLite Deployment</p>
            <p style="color:#2E7D32; font-size:16px; font-weight:bold; font-style:italic; margin:0;">By Kang Adit</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Area Dropzone Upload Gambar
st.subheader("📸 Unggah Gambar Sampah")
uploaded_images = st.file_uploader(
    "Seret dan lepaskan satu atau beberapa gambar sampah di sini (Format: JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Batasan Nama Kelas Dataset RealWaste
class_names = ['Cardboard', 'Food Waste', 'Glass', 'Metal', 'Miscellaneous', 'Paper', 'Plastic', 'Textiles', 'Vegetation']

# Memuat Model TFLite dengan Caching agar Web Cepat
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="model_waste.tflite") 
    return interpreter

interpreter = load_model()
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- PROSES PREDIKSI DAN GRID DISPLAY ---
if uploaded_images:
    st.divider()
    st.markdown("<h3 style='color:#2E7D32;'>📊 Hasil Analisis AI RealWaste</h3>", unsafe_allow_html=True)
    
    # Membuat grid kolom dinamis (maksimal 3 kolom per baris)
    num_cols = 3
    for i in range(0, len(uploaded_images), num_cols):
        batch_images = uploaded_images[i:i+num_cols]
        cols = st.columns(len(batch_images))
        
        for idx, uploaded_image in enumerate(batch_images):
            # Membuka gambar
            image = Image.open(uploaded_image).convert("RGB")
            
            # 1. KUNCI UKURAN: Paksa gambar di-resize ke kotak sempurna (300x300) agar simetris di web
            web_display_img = image.resize((300, 300), Image.Resampling.LANCZOS)
            
            # Pra-pemrosesan Gambar untuk Model AI (tetap 224x224 sesuai arsitektur model)
            resized_img = image.resize((224, 224))
            img_array = np.array(resized_img, dtype=np.float32)
            input_data = np.expand_dims(img_array, axis=0)
            
            # Menjalankan Prediksi Model
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Mengambil hasil skor tertinggi
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_class = class_names[output_data.argmax()]
            
            # Menampilkan ke dalam Grid Kolom
            with cols[idx]:
                # Membuka container box HTML utama (Bungkus semua jadi satu kesatuan)
                st.markdown(f"""
                    <div style="background-color:#FFFFFF; padding:15px; border-radius:12px; border:1px solid #E0E0E0; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); text-align:center; margin-bottom: 20px;">
                        <span style="font-weight:bold; color:#424242; font-size:16px; display:block; margin-bottom:10px;">📦 Sampah Ke-{i + idx + 1}</span>
                """, unsafe_allow_html=True)
                
                # 2. TAMPILKAN GAMBAR: Menggunakan gambar yang sudah disamakan rasionya
                st.image(web_display_img, use_container_width=True)
                
                # Menutup container box HTML dan menyatukan dengan Label Hasil di dalamnya
                st.markdown(f"""
                        <div style="background-color:#2E7D32; padding:10px; border-radius:8px; text-align:center; margin-top:12px;">
                            <span style="color:white; font-weight:bold; font-size:15px;">👉 Kategori: {pred_class}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
else:
    # Tampilan petunjuk jika user belum mengupload gambar apapun
    st.info("💡 Petunjuk: Silakan unggah satu atau beberapa foto sampah di atas untuk melihat bagaimana AI mendeteksi kategorinya secara otomatis.")