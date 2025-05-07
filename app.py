# -*- coding: utf-8 -*-
"""
Backend Flask untuk Sistem Deteksi Objek Kendaraan Sederhana.

Menangani upload gambar, menjalankan inferensi model Deep Learning,
menggambar hasil, dan menyajikan ke antarmuka web.
"""

# --- Mengimpor Library ---
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import os
import time # Untuk memberi nama unik pada file gambar hasil

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename # Untuk mengamankan nama file yang diunggah

# --- Konfigurasi Flask ---
app = Flask(__name__)

# Konfigurasi folder untuk menyimpan gambar yang diunggah dan gambar hasil sementara
# Folder 'static' akan diakses oleh web server untuk menampilkan gambar
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Pastikan folder static ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Memuat Model Deteksi Objek Pre-trained ---
# Model akan dimuat sekali saat aplikasi Flask dimulai.
# Menggunakan model SSD MobileNet V2 FPNLite dari TensorFlow Hub.
model = None # Variabel global untuk menyimpan model

def load_detection_model():
    """Memuat model deteksi objek dari TensorFlow Hub."""
    global model # Mengakses variabel model global
    model_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
    print(f"\n--> Memuat model pre-trained dari TensorFlow Hub: {model_handle}...")
    try:
        model = hub.load(model_handle)
        print("--> Model berhasil dimuat.")
    except Exception as e:
        print(f"Error: Gagal memuat model dari TensorFlow Hub. Pastikan koneksi internet aktif dan URL model benar. Error: {e}")
        model = None # Set model ke None jika gagal dimuat

# Panggil fungsi untuk memuat model saat startup aplikasi
with app.app_context():
    load_detection_model()


# --- Fungsi Bantuan (Menggambar Bounding Box) ---
# Fungsi ini sama dengan yang sebelumnya, untuk menggambar box dan label.
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               class_name,
                               score,
                               color=(0, 255, 0), # Warna default box dan teks (Hijau) dalam format BGR
                               font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=0.6, # Ukuran skala font
                               font_thickness=2): # Ketebalan garis/font
    """
    Menggambar kotak pembatas dan label pada gambar menggunakan OpenCV.
    Koordinat box (ymin, xmin, ymax, xmax) diasumsikan relatif (0-1).
    """
    # Pastikan image adalah NumPy array
    if not isinstance(image, np.ndarray):
         print("Warning: Input image for drawing is not a NumPy array.")
         return image # Kembalikan image asli jika tipe data salah

    image_height, image_width, _ = image.shape
    # Konversi koordinat relatif (0-1) ke piksel
    (left, right, top, bottom) = (xmin * image_width, xmax * image_width,
                                  ymin * image_height, ymax * image_height)

    # Pastikan koordinat berada di dalam batas gambar (membulatkan ke integer)
    left, right, top, bottom = max(0, int(left)), min(image_width, int(right)), max(0, int(top)), min(image_height, int(bottom))

    # --- Gambar Bounding Box ---
    # Warna di OpenCV adalah BGR, jadi (0, 255, 0) adalah hijau.
    cv2.rectangle(image, (left, top), (right, bottom), color, font_thickness)

    # --- Tulis Label dan Skor ---
    label = f"{class_name}: {score:.2f}"

    # Mendapatkan ukuran teks label dan baseline menggunakan cv2.getTextSize
    (label_width, label_height), baseLine = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Tentukan posisi Y untuk teks (di atas bounding box)
    text_origin_y = max(label_height + 5, top - 5)
    text_origin_x = left
    text_origin = (text_origin_x, text_origin_y)

    # Tulis teks label
    cv2.putText(image, label, text_origin,
                font, font_scale, color, font_thickness)

    return image

# --- Mapping ID Kelas COCO ke Nama Kelas ---
# Dictionary ini memetakan ID kelas COCO ke nama kelas.
category_map_coco = {
    1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane',
    6: 'Bus', 7: 'Train', 8: 'Truck', 9: 'Boat', 10: 'Traffic Light', 11: 'Fire Hydrant',
    13: 'Stop Sign', 14: 'Parking Meter', 15: 'Bench', 16: 'Bird', 17: 'Cat',
    18: 'Dog', 19: 'Horse', 20: 'Sheep', 21: 'Cow', 22: 'Elephant', 23: 'Bear',
    24: 'Zebra', 25: 'Giraffe', 27: 'Backpack', 28: 'Umbrella', 31: 'Handbag',
    32: 'Tie', 33: 'Suitcase', 34: 'Frisbee', 35: 'Skis', 36: 'Snowboard',
    37: 'Sports Ball', 38: 'Kite', 39: 'Baseball Bat', 40: 'Baseball Glove',
    41: 'Skateboard', 42: 'Surfboard', 43: 'Tennis Racket', 44: 'Bottle',
    46: 'Wine Glass', 47: 'Cup', 48: 'Fork', 49: 'Knife', 50: 'Spoon', 51: 'Bowl',
    52: 'Banana', 53: 'Apple', 54: 'Sandwich', 55: 'Orange', 56: 'Broccoli',
    57: 'Carrot', 58: 'Hot Dog', 59: 'Pizza', 60: 'Donut', 61: 'Cake',
    62: 'Chair', 63: 'Couch', 64: 'Potted Plant', 65: 'Bed', 67: 'Dining Table',
    70: 'Toilet', 72: 'TV', 73: 'Laptop', 74: 'Mouse', 75: 'Remote', 76: 'Keyboard',
    77: 'Cell Phone', 78: 'Microwave', 79: 'Oven', 80: 'Toaster', 81: 'Sink',
    82: 'Refrigerator', 84: 'Book', 85: 'Clock', 86: 'Vase', 87: 'Scissors',
    88: 'Teddy Bear', 89: 'Hair Dryer', 90: 'Toothbrush',
}

# --- Logic Inferensi Deteksi Objek ---
def run_inference(image_path, detection_model):
    """
    Memuat gambar, menjalankan model deteksi, memproses hasil, dan menggambar box.
    Mengembalikan path gambar hasil dan daftar deteksi.
    """
    if detection_model is None:
        print("Error: Model deteksi belum dimuat.")
        return None, []

    # Membaca gambar menggunakan OpenCV
    image_np = cv2.imread(image_path)

    if image_np is None:
        print(f"Error: Gagal memuat gambar dari {image_path}")
        return None, []

    # OpenCV membaca gambar dalam format BGR, model DL butuh RGB.
    image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Copy gambar asli dalam format BGR untuk digambar nanti oleh OpenCV
    image_np_bgr_drawn = np.copy(image_np)

    # Model dari TF Hub mengharapkan input berupa batch tensor TensorFlow, tipe data uint8.
    input_tensor = tf.convert_to_tensor(image_np_rgb, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...] # Tambahkan dimensi 'batch'

    print("\n--> Menjalankan inferensi (prediksi deteksi objek)...")
    detections = detection_model(input_tensor)
    print("--> Inferensi selesai.")

    # Memproses Hasil Deteksi
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.uint32)
    scores = detections['detection_scores'][0].numpy()
    num_detections = int(detections['num_detections'][0].numpy())

    # Filtering Deteksi
    min_confidence_threshold = 0.5 # Ambil deteksi yang minimal 50% yakin

    filtered_detections_list = [] # List untuk menyimpan deteksi yang difilter

    print(f"--> Ditemukan {num_detections} deteksi total.")
    print(f"--> Deteksi yang melewati threshold ({min_confidence_threshold:.2f}):")

    # Loop melalui semua deteksi yang valid
    for i in range(num_detections):
        # Jika skor deteksi saat ini di atas threshold...
        if scores[i] > min_confidence_threshold:
            class_id = classes[i]
            class_name = category_map_coco.get(class_id, f'ID: {class_id}')
            score = scores[i]
            ymin, xmin, ymax, xmax = boxes[i]

            # Simpan deteksi yang difilter
            filtered_detections_list.append({
                'class_id': class_id,
                'class_name': class_name,
                'score': float(score), # Ubah ke float agar mudah di-serialize jika perlu
                'ymin': float(ymin), 'xmin': float(xmin), 'ymax': float(ymax), 'xmax': float(xmax) # Ubah ke float
            })

            # Gambar bounding box pada gambar BGR asli
            image_np_bgr_drawn = draw_bounding_box_on_image(
                image_np_bgr_drawn,
                ymin, xmin, ymax, xmax,
                class_name, score,
                color=(0, 255, 0), # Warna Hijau
                font_scale=0.6,
                font_thickness=2
            )
            print(f"  - Objek: {class_name} (ID COCO: {class_id}), Skor: {score:.2f}")

    # --- Menyimpan Gambar Hasil ---
    # Buat nama file unik untuk gambar hasil
    timestamp = int(time.time())
    output_filename = f"detection_result_{timestamp}.jpg"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    # Simpan gambar hasil (dalam format BGR karena OpenCV menyimpannya seperti itu)
    cv2.imwrite(output_path, image_np_bgr_drawn)
    print(f"--> Gambar hasil deteksi disimpan di: {output_path}")

    # Mengembalikan path gambar hasil (relatif terhadap folder static) dan daftar deteksi
    return f'/{app.config["UPLOAD_FOLDER"]}/{output_filename}', filtered_detections_list


# --- Rute Flask ---

@app.route('/')
def index():
    """Menampilkan halaman utama (form upload)."""
    # Render template index.html tanpa hasil deteksi awal
    return render_template('index.html', image_path=None, detections=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Menangani request upload gambar dan menjalankan deteksi."""
    # Cek apakah model berhasil dimuat saat startup
    if model is None:
        return "Error: Model deteksi gagal dimuat saat startup.", 500 # Kode status 500 Internal Server Error

    # Cek apakah request memiliki bagian file
    if 'image_file' not in request.files:
        return "Tidak ada bagian file 'image_file' dalam request.", 400 # Kode status 400 Bad Request

    file = request.files['image_file']

    # Cek apakah file kosong
    if file.filename == '':
        return "Tidak ada file yang dipilih.", 400

    # Cek apakah tipe file diizinkan (opsional, bisa ditambahkan validasi lebih ketat)
    # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    # if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
    #     return "Tipe file tidak diizinkan.", 400

    if file:
        # Amankan nama file untuk mencegah serangan direktori traversal
        filename = secure_filename(file.filename)
        # Buat path lengkap untuk menyimpan file yang diunggah sementara
        # Kita bisa langsung memproses dari stream file tanpa menyimpan,
        # tetapi menyimpan sementara memudahkan debugging dan penggunaan OpenCV.
        # Untuk kesederhanaan, kita simpan sementara di folder static juga,
        # tapi sebaiknya gunakan folder temp terpisah di aplikasi nyata.
        temp_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{int(time.time())}_{filename}")
        file.save(temp_upload_path)
        print(f"\n--> File diunggah dan disimpan sementara di: {temp_upload_path}")

        # Jalankan inferensi deteksi objek pada gambar yang diunggah
        output_image_path, detected_objects = run_inference(temp_upload_path, model)

        # Hapus file yang diunggah sementara setelah diproses (opsional tapi disarankan)
        try:
            os.remove(temp_upload_path)
            print(f"--> File sementara {temp_upload_path} dihapus.")
        except OSError as e:
            print(f"Error menghapus file sementara {temp_upload_path}: {e}")


        if output_image_path:
            # Render template index.html lagi, kali ini dengan path gambar hasil dan deteksi
            return render_template('index.html', image_path=output_image_path, detections=detected_objects)
        else:
            return "Gagal memproses gambar untuk deteksi.", 500

    return "Terjadi kesalahan saat mengunggah file.", 500 # Default error jika ada masalah lain


# --- Menjalankan Aplikasi Flask ---
if __name__ == '__main__':
    # Menjalankan web server Flask.
    # debug=True akan memberikan informasi error yang membantu saat pengembangan.
    print("\n--- Menjalankan Aplikasi Flask ---")
    print("Buka browser dan akses: http://127.0.0.1:5000/")
    # Pastikan Anda berada di dalam virtual environment saat menjalankan file ini.
    # Pastikan juga model berhasil dimuat saat startup.
    app.run(debug=True)

