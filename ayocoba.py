import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import base64
from io import BytesIO
import time
from collections import Counter

# Mengatur konfigurasi halaman agar responsif dan modern
st.set_page_config(
    page_title="RealWaste AI Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi pembersih HTML dari leading whitespaces agar Streamlit/Markdown tidak mengiranya sebagai code block
def clean_html(html_str):
    return "\n".join([line.strip() for line in html_str.split("\n")])

# Pemetaan metadata kategori (Ikon, Warna Primer, & Warna Background Lencana)
category_meta = {
    'Cardboard': {'icon': '📦', 'color': '#B45309', 'bg': '#FEF3C7'},
    'Food Waste': {'icon': '🍎', 'color': '#C2410C', 'bg': '#FFEDD5'},
    'Glass': {'icon': '🍷', 'color': '#0369A1', 'bg': '#E0F2FE'},
    'Metal': {'icon': '🔩', 'color': '#475569', 'bg': '#F1F5F9'},
    'Miscellaneous': {'icon': '🌀', 'color': '#6D28D9', 'bg': '#EDE9FE'},
    'Paper': {'icon': '📄', 'color': '#1E3A8A', 'bg': '#DBEAFE'},
    'Plastic': {'icon': '🥤', 'color': '#0891B2', 'bg': '#ECFEFF'},
    'Textiles': {'icon': '👕', 'color': '#BE185D', 'bg': '#FCE7F3'},
    'Vegetation': {'icon': '🌿', 'color': '#047857', 'bg': '#D1FAE5'},
}

# Kamus Panduan Edukasi Daur Ulang Dinamis
eco_tips = {
    'Cardboard': 'Bilas sisa minyak, lipat rata kardus agar menghemat ruang pembuangan, dan jaga tetap kering.',
    'Food Waste': 'Sangat baik diolah menjadi pupuk kompos tanaman organik atau pakan biokonversi maggot BSF.',
    'Glass': 'Pisahkan dari botol plastik, bilas bersih, dan buang dalam wadah aman untuk didaur ulang tanpa batas waktu.',
    'Metal': 'Bilas sisa makanan/minuman, remas kaleng hingga pipih agar memudahkan pengangkutan daur ulang.',
    'Miscellaneous': 'Sampah residu/campuran. Buang ke TPA atau pisahkan secara spesifik sesuai panduan pembuangan lokal.',
    'Paper': 'Jaga kertas agar tidak basah atau berminyak supaya serat kertas tetap berkualitas tinggi saat didaur ulang.',
    'Plastic': 'Bilas wadah plastik, remas untuk mengurangi volume sampah, lalu salurkan melalui bank sampah terdekat.',
    'Textiles': 'Pakaian layak pakai sebaiknya didonasikan, sedangkan kain rusak bisa dijadikan lap pembersih rumah tangga.',
    'Vegetation': 'Bahan organik hijau ini sangat baik dicampur dengan sampah cokelat kering untuk pembuatan kompos padat.',
}

# Fungsi pembantu untuk mengonversi gambar PIL ke base64 string agar bisa dipacking dalam HTML card secara aman
def get_image_base64(pil_img):
    buffered = BytesIO()
    # Simpan sebagai PNG berkualitas tinggi
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- CSS CUSTOM UNTUK OVERHAUL TAMPILAN GAYA DEWA ---
st.markdown(clean_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Overrides */
    .stApp, html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #F8FAFC !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #10B981 !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 20px !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #94A3B8 !important;
    }
    
    /* Divider Custom Styling */
    hr {
        margin: 1.5rem 0 !important;
        border-color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }
    
    /* Hero Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #064E3B 0%, #059669 60%, #10B981 100%);
        padding: 35px 40px;
        border-radius: 24px;
        margin-bottom: 35px;
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.35);
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1.2px;
        color: white !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.12);
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.85);
        margin: 8px 0 0 0;
        font-weight: 400;
    }
    .hero-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 24px;
        border-top: 1px solid rgba(255,255,255,0.15);
        padding-top: 20px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.18);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .hero-author {
        font-size: 14px;
        font-weight: 700;
        color: #A7F3D0;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Category Pills Sidebar */
    .category-container {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
        margin-top: 15px;
    }
    .category-badge-sidebar {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 600;
        background-color: #1E293B;
        border: 1px solid #334155;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }
    .category-badge-sidebar:hover {
        transform: translateX(4px);
        border-color: #10B981;
        background-color: #161F30;
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        background-color: white !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 20px !important;
        padding: 30px 20px !important;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #10B981 !important;
        box-shadow: 0 12px 20px -8px rgba(16, 185, 129, 0.15) !important;
    }
    
    /* Summary Dashboard */
    .summary-dashboard {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }
    .summary-card {
        background-color: white;
        padding: 18px 24px;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .summary-card:hover {
        transform: translateY(-4px);
        border-color: #10B981;
        box-shadow: 0 12px 20px -8px rgba(16, 185, 129, 0.15);
    }
    .summary-val {
        display: block;
        font-size: 22px;
        font-weight: 800;
        color: #064E3B;
        margin-bottom: 4px;
    }
    .summary-lbl {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Custom Result Card */
    .result-card {
        background-color: white;
        padding: 22px;
        border-radius: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 28px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        position: relative;
    }
    .result-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 30px -10px rgba(16, 185, 129, 0.12);
        border-color: #10B981;
    }
    .result-card-header {
        font-weight: 800;
        color: #94A3B8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    .result-image-wrapper {
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 18px;
        border: 1px solid #F1F5F9;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Dynamic Category Badge */
    .badge-result {
        padding: 12px 16px;
        border-radius: 14px;
        font-size: 15px;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        margin-top: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        letter-spacing: -0.2px;
    }
    
    /* Accuracy Section */
    .accuracy-container {
        margin-top: 16px;
        text-align: left;
        border-top: 1px solid #F1F5F9;
        padding-top: 14px;
    }
    .accuracy-header {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        margin-bottom: 6px;
    }
    .accuracy-bar-bg {
        background-color: #E2E8F0;
        height: 8px;
        border-radius: 10px;
        overflow: hidden;
    }
    .accuracy-bar-fill {
        background: linear-gradient(90deg, #059669 0%, #10B981 100%);
        height: 100%;
        border-radius: 10px;
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
    }
    
    /* Eco-Tip Box styling */
    .eco-tip-box {
        background-color: #F8FAFC;
        padding: 12px;
        border-radius: 12px;
        text-align: left;
        margin-top: 16px;
        border-left: 3px solid #E2E8F0;
    }
    
    /* Alert customization */
    .stAlert {
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
    }
</style>
"""), unsafe_allow_html=True)

# --- SIDEBAR (SISI KIRI WEB) ---
with st.sidebar:
    st.markdown("<h2>🌱 Pusat Informasi</h2>", unsafe_allow_html=True)
    st.write("Aplikasi pintar bertenaga **Kecerdasan Buatan (AI)** yang dirancang untuk mendeteksi 9 kategori jenis sampah secara instan demi mewujudkan gerakan *Eco-Friendly*.")
    
    st.divider()
    st.markdown("<h4 style='color: #F8FAFC; margin-bottom: 12px; font-weight: 700;'>📋 9 Kategori yang Didukung:</h4>", unsafe_allow_html=True)
    
    # Pill badges dinamis untuk kategori di sidebar
    sidebar_badges_html = """
    <div class="category-container">
    """
    for cls_name, meta in category_meta.items():
        sidebar_badges_html += f"""
        <div class="category-badge-sidebar" style="border-left: 3px solid {meta['color']};">
            <span style="font-size: 16px;">{meta['icon']}</span>
            <span style="color: #E2E8F0 !important;">{cls_name}</span>
        </div>
        """
    sidebar_badges_html += "</div>"
    st.markdown(clean_html(sidebar_badges_html), unsafe_allow_html=True)
    
    st.divider()
    
    # Penanda hak cipta di sidebar kiri
    st.markdown(clean_html("""
        <div style="text-align: center; padding: 10px 0;">
            <p style="color: #94A3B8; font-size: 13px; margin: 0;">🛠️ Developed by <b style="color: #10B981;">Kang Adit</b></p>
            <p style="color: #64748B; font-size: 11px; margin: 2px 0 0 0;">Tugas Praktikum Machine Learning v2.0</p>
        </div>
    """), unsafe_allow_html=True)

# --- HALAMAN UTAMA ---
# Desain Header Atas menggunakan Gradasi Elegan
st.markdown(clean_html("""
    <div class="hero-container">
        <h1 class="hero-title">♻️ Sistem Klasifikasi Sampah Otomatis</h1>
        <p class="hero-subtitle">Mendeteksi bahan daur ulang secara instan menggunakan arsitektur Deep Learning TFLite</p>
        <div class="hero-meta-row">
            <div style="display: flex; gap: 8px;">
                <span class="hero-badge">📦 Dataset: RealWaste</span>
                <span class="hero-badge">⚡ TFLite Optimized</span>
            </div>
            <span class="hero-author">👤 By Kang Adit</span>
        </div>
    </div>
"""), unsafe_allow_html=True)

# Area Dropzone Upload Gambar
st.markdown("<h3 style='color: #064E3B; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 15px;'>📸 Unggah Gambar Sampah</h3>", unsafe_allow_html=True)
uploaded_images = st.file_uploader(
    "Seret dan lepaskan satu atau beberapa gambar sampah di sini (Format: JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
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
    
    # -------------------------------------------------------------------------
    # TAHAP PRE-PROCESSING & BATCH PREDICTION (Untuk Keperluan Statistik Rangkuman)
    # -------------------------------------------------------------------------
    results_list = []
    
    for idx, uploaded_image in enumerate(uploaded_images):
        image = Image.open(uploaded_image).convert("RGB")
        
        # KUNCI UKURAN: Paksa gambar di-resize ke kotak sempurna (300x300) agar simetris di web
        web_display_img = image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Mulai Hitung Waktu Inferensi
        start_time = time.perf_counter()
        
        # Pra-pemrosesan Gambar untuk Model AI (tetap 224x224 sesuai arsitektur model)
        resized_img = image.resize((224, 224))
        img_array = np.array(resized_img, dtype=np.float32)
        input_data = np.expand_dims(img_array, axis=0)
        
        # Menjalankan Prediksi Model TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Mengambil hasil skor tertinggi dan hitung keyakinan (confidence)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_idx = output_data.argmax()
        pred_class = class_names[pred_idx]
        
        # Selesai Hitung Waktu Inferensi
        inference_time = (time.perf_counter() - start_time) * 1000 # dalam milidetik
        
        # Menghitung confidence score dengan Softmax yang stabil secara matematis
        scores = output_data[0]
        if np.issubdtype(scores.dtype, np.integer):
            scores = scores.astype(np.float32)
        
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        confidence = probs[pred_idx] * 100
        
        # Konversi image ke base64
        img_base64 = get_image_base64(web_display_img)
        
        # Simpan metadata hasil ke list
        results_list.append({
            'index': idx + 1,
            'pred_class': pred_class,
            'confidence': confidence,
            'inference_time': inference_time,
            'img_base64': img_base64
        })
    
    # -------------------------------------------------------------------------
    # PERHITUNGAN STATISTIK RINGKASAN (DASHBOARD)
    # -------------------------------------------------------------------------
    total_uploaded = len(results_list)
    avg_confidence = np.mean([item['confidence'] for item in results_list])
    avg_inference = np.mean([item['inference_time'] for item in results_list])
    
    # Mencari mayoritas/kategori dominan
    classes_list = [item['pred_class'] for item in results_list]
    dominant_class = Counter(classes_list).most_common(1)[0][0]
    dominant_icon = category_meta.get(dominant_class, {'icon': '🗑️'})['icon']
    
    # Tampilkan Header Dashboard Rangkuman
    st.markdown("<h3 style='color: #064E3B; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 20px;'>📊 Rangkuman Statistik Analisis AI</h3>", unsafe_allow_html=True)
    
    st.markdown(clean_html(f"""
        <div class="summary-dashboard">
            <div class="summary-card">
                <span class="summary-val">📸 {total_uploaded} Foto</span>
                <span class="summary-lbl">Total Sampah</span>
            </div>
            <div class="summary-card">
                <span class="summary-val">{dominant_icon} {dominant_class}</span>
                <span class="summary-lbl">Mayoritas Jenis</span>
            </div>
            <div class="summary-card">
                <span class="summary-val">🟢 {avg_confidence:.1f}%</span>
                <span class="summary-lbl">Rata-rata Akurasi</span>
            </div>
            <div class="summary-card">
                <span class="summary-val">⚡ {avg_inference:.1f} ms</span>
                <span class="summary-lbl">Rata-rata Proses</span>
            </div>
        </div>
    """), unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # RENDER GRID HASIL DETEKSI AI
    # -------------------------------------------------------------------------
    st.markdown("<h3 style='color: #064E3B; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 20px;'>🔍 Detail Hasil Klasifikasi</h3>", unsafe_allow_html=True)
    
    # Membuat grid kolom dinamis (maksimal 3 kolom per baris)
    num_cols = 3
    for i in range(0, len(results_list), num_cols):
        batch_results = results_list[i:i+num_cols]
        cols = st.columns(len(batch_results))
        
        for idx, result in enumerate(batch_results):
            pred_class = result['pred_class']
            confidence = result['confidence']
            inference_time = result['inference_time']
            img_base64 = result['img_base64']
            
            # Ambil meta visual untuk lencana dinamis & panduan eco-tip
            meta = category_meta.get(pred_class, {'icon': '🗑️', 'color': '#64748B', 'bg': '#F1F5F9'})
            tip_text = eco_tips.get(pred_class, 'Daur ulang dan buang sesuai instruksi daur ulang setempat.')
            
            # Menampilkan ke dalam Grid Kolom
            with cols[idx]:
                st.markdown(clean_html(f"""
                    <div class="result-card">
                        <div>
                            <span class="result-card-header">
                                <span>♻️ SAMPAH KE-{result['index']}</span>
                                <span style="font-weight: 600; text-transform: none; color: #10B981; letter-spacing: 0;">⚡ {inference_time:.1f} ms</span>
                            </span>
                            <div class="result-image-wrapper">
                                <img src="{img_base64}" style="width:100%; aspect-ratio:1/1; object-fit:cover;" />
                            </div>
                        </div>
                        <div>
                            <div class="badge-result" style="background-color: {meta['bg']}; color: {meta['color']}; border: 1px solid {meta['color']}33;">
                                <span style="font-size: 20px;">{meta['icon']}</span>
                                <span>{pred_class}</span>
                            </div>
                            <div class="accuracy-container">
                                <div class="accuracy-header">
                                    <span>Akurasi Deteksi</span>
                                    <span>{confidence:.1f}%</span>
                                </div>
                                <div class="accuracy-bar-bg">
                                    <div class="accuracy-bar-fill" style="width: {confidence}%;"></div>
                                </div>
                            </div>
                            <div class="eco-tip-box" style="border-left: 3px solid {meta['color']};">
                                <span style="font-weight: 800; font-size: 11px; color: {meta['color']}; text-transform: uppercase; display: block; margin-bottom: 4px;">💡 Eco-Tip Daur Ulang</span>
                                <span style="font-size: 12px; color: #475569; display: block; line-height: 1.4;">{tip_text}</span>
                            </div>
                        </div>
                    </div>
                """), unsafe_allow_html=True)
else:
    # Tampilan petunjuk jika user belum mengupload gambar apapun
    st.info("💡 Petunjuk: Silakan unggah satu atau beberapa foto sampah di atas untuk melihat bagaimana AI mendeteksi kategorinya secara otomatis.")