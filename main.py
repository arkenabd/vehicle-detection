# -*- coding: utf-8 -*-
"""
Kode Python Lengkap untuk Demonstrasi Deteksi Objek Kendaraan (Inferensi).

Kode ini menggunakan model Deep Learning pre-trained dari TensorFlow Hub
untuk mendeteksi objek dalam sebuah gambar dari dataset lokal Anda.

Langkah-langkah dalam kode ini mencakup:
1. Memuat library yang dibutuhkan.
2. Mendefinisikan fungsi bantuan untuk menggambar hasil deteksi.
3. Memuat model deteksi objek pre-trained.
4. Memuat gambar input dari lokasi spesifik di komputer Anda.
5. Menjalankan proses inferensi (prediksi) menggunakan model.
6. Memproses hasil deteksi (box, kelas, skor).
7. Menggambar bounding box dan label pada gambar.
8. Menampilkan gambar hasil deteksi.

Anda dapat menjalankan file ini langsung di VSCode.
"""

# --- Langkah 4.1: Persiapan Lingkungan dan Library ---
# Mengimpor library Python yang dibutuhkan.
# Pastikan library ini sudah terinstal: pip install tensorflow opencv-python matplotlib tensorflow_hub
import tensorflow as tf             # Framework Deep Learning utama (TensorFlow)
import numpy as np                  # Untuk komputasi numerik dan manipulasi array (NumPy)
import cv2                          # Library Computer Vision (OpenCV) untuk baca/tulis gambar dan visualisasi
import matplotlib.pyplot as plt     # Untuk menampilkan gambar hasil (Matplotlib)
import tensorflow_hub as hub        # Untuk memuat model pre-trained dari TensorFlow Hub
import os                           # Modul OS untuk manipulasi path file

# Memeriksa versi TensorFlow (opsional, untuk debugging)
# print(f"Versi TensorFlow: {tf.__version__}")
# Memeriksa apakah TensorFlow melihat GPU (opsional, untuk debugging)
# print(f"Nomor GPU yang tersedia: {len(tf.config.list_physical_devices('GPU'))}")


# --- Fungsi Bantuan (Berguna untuk Menggambar) ---
# Fungsi ini membantu menggambar bounding box dan label pada gambar menggunakan OpenCV.
# Kode fungsi ini sudah diperbaiki untuk mengatasi NameError 'font_size'.
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
    image_height, image_width, _ = image.shape
    # Konversi koordinat relatif (0-1) ke piksel
    (left, right, top, bottom) = (xmin * image_width, xmax * image_width,
                                  ymin * image_height, ymax * image_height)

    # Pastikan koordinat berada di dalam batas gambar (membulatkan ke integer)
    left, right, top, bottom = max(0, int(left)), min(image_width, int(right)), max(0, int(top)), min(image_height, int(bottom))

    # --- Gambar Bounding Box ---
    # cv2.rectangle(gambar, titik_kiri_atas, titik_kanan_bawah, warna, tebal_garis)
    # Warna di OpenCV adalah BGR, jadi (0, 255, 0) adalah hijau.
    cv2.rectangle(image, (left, top), (right, bottom), color, font_thickness)

    # --- Tulis Label dan Skor ---
    label = f"{class_name}: {score:.2f}"

    # Mendapatkan ukuran teks label dan baseline menggunakan cv2.getTextSize
    # Ini diperlukan untuk menentukan posisi teks agar tidak tumpang tindih atau keluar gambar.
    (label_width, label_height), baseLine = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Tentukan posisi Y untuk teks (di atas bounding box)
    # Ambil posisi Y dari tepi atas box (top), kurangi tinggi teks + margin.
    # Pastikan posisi Y tidak negatif (minimal di 0)
    text_origin_y = max(label_height + 5, top - 5) # Posisi Y: 5 piksel di atas box atau setinggi teks + margin jika box di tepi atas gambar
    text_origin_x = left # Posisi X sama dengan tepi kiri box
    text_origin = (text_origin_x, text_origin_y) # Koordinat (X, Y) untuk meletakkan teks

    # (Opsional) Gambar latar belakang untuk teks agar lebih mudah dibaca
    # Hitung koordinat latar belakang teks: (kiri_atas X, kiri_atas Y), (kanan_bawah X, kanan_bawah Y)
    # bg_top_left = (left, text_origin_y - label_height - 2) # 2 piksel di atas teks
    # bg_bottom_right = (left + label_width, text_origin_y + baseLine + 2) # 2 piksel di bawah baseline
    # cv2.rectangle(image, bg_top_left, bg_bottom_right, color, cv2.FILLED) # Warna latar sama dgn box

    # Tulis teks label: cv2.putText(gambar, teks, posisi, font, skala, warna, tebal_garis)
    # Warna teks di sini dibuat sama dengan warna box (hijau). Jika pakai latar, warna teks bisa hitam (0,0,0).
    cv2.putText(image, label, text_origin,
                font, font_scale, color, font_thickness)

    return image


# --- Bagian Utama Script ---
if __name__ == "__main__":
    # Kode di dalam blok ini akan dijalankan ketika script dieksekusi langsung

    print("--- Memulai Skrip Deteksi Objek Kendaraan (Inferensi) ---")

    # --- Langkah 4.2: Memuat Model Deteksi Objek Pre-trained ---
    # Kita menggunakan model Deteksi Objek yang sudah dilatih di dataset besar seperti COCO.
    # Dataset COCO mencakup banyak kategori objek, termasuk berbagai jenis kendaraan.
    # Model ini akan kita gunakan untuk MENDETEKSI objek pada gambar dari dataset Anda.
    model_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1" # Contoh model ringan & cepat dari TF Hub

    print(f"\n--> Memuat model pre-trained dari TensorFlow Hub: {model_handle}...")
    try:
        # Memuat model dari URL. Proses ini mungkin butuh koneksi internet saat pertama kali dijalankan
        # karena model akan diunduh ke cache lokal (~50MB untuk model ini).
        model = hub.load(model_handle)
        print("--> Model berhasil dimuat.")
        model_loaded = True
    except Exception as e:
        print(f"Error: Gagal memuat model dari TensorFlow Hub. Pastikan koneksi internet aktif dan URL model benar.")
        print(f"Detail Error: {e}")
        model_loaded = False


    if model_loaded:
        # --- Langkah 4.3: Memuat dan Mempersiapkan Gambar Input ---
        # Kita akan memuat sebuah gambar dari folder dataset deteksi objek kendaraan lokal Anda.
        # Dataset Anda adalah contoh dari jenis data yang digunakan untuk melatih model deteksi objek.
        # Kita gunakan salah satu gambar dari dataset Anda di sini untuk MENGUJI model pre-trained COCO.

        # GANTI path di bawah sesuai dengan lokasi folder 'train' dataset Anda di komputer Anda!
        # Disarankan menggunakan garis miring maju (/) di path agar lebih kompatibel.
        dataset_train_base_path = 'D:/explore_python/compvis/uts/dataset_object_detection_vehicle/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.tensorflow/test' # <<< GANTI JIKA PATH BERBEDA

        # Nama salah satu file gambar DARI DALAM folder dataset_train_base_path.
        # Anda bisa lihat nama file yang persis di File Explorer Windows.
        # GANTI string di bawah dengan nama file gambar yang benar-benar ADA di folder tersebut.
        image_file_name = 'frame_6406_jpg.rf.ff63b851413c468025e0f9f2dad80fd0.jpg' # <<< GANTI DENGAN NAMA FILE GAMBAR YANG ADA DI DATASET ANDA

        # Menggabungkan path folder dan nama file gambar menggunakan os.path.join
        # Ini cara yang aman untuk menggabungkan path lintas sistem operasi.
        # image_full_path = os.path.join(dataset_train_base_path, image_file_name)
        image_full_path='D:/explore_python/compvis/uts/test_compvis_001.png'

        # *** DEBUGGING STEP (Opsional): Cetak path lengkap untuk verifikasi ***
        # print(f"Jalur gambar yang dicoba dimuat: {image_full_path}")
        # Bandingkan output print ini dengan path asli file di File Explorer Windows Anda.

        print(f"\n--> Memuat gambar dari: {image_full_path}")

        # Membaca gambar menggunakan OpenCV (cv2).
        # cv2.imread mengembalikan None jika gambar tidak ditemukan atau gagal dibaca.
        image_np = cv2.imread(image_full_path)

        # Cek apakah gambar berhasil dimuat
        if image_np is None:
            print(f"Error: Gambar tidak ditemukan atau gagal dimuat dari {image_full_path}")
            print("Pastikan path dan nama file gambar di Langkah 4.3 sudah benar dan file tidak rusak.")
            image_loaded = False
        else:
            print("--> Gambar berhasil dimuat.")
            image_loaded = True

            # OpenCV membaca gambar dalam format BGR (Blue, Green, Red) secara default.
            # Model Deep Learning umum (terutama yang dilatih di dataset seperti COCO/ImageNet)
            # mengharapkan input dalam format RGB (Red, Green, Blue).
            image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Model dari TF Hub mengharapkan input berupa batch tensor TensorFlow, tipe data uint8.
            # Ubah numpy array gambar (dalam format RGB) menjadi tensor TensorFlow.
            input_tensor = tf.convert_to_tensor(image_np_rgb, dtype=tf.uint8)

            # Tambahkan dimensi baru di depan untuk merepresentasikan 'batch size'.
            # Karena kita hanya memproses satu gambar, batch size-nya adalah 1.
            input_tensor = input_tensor[tf.newaxis, ...] # Bentuk tensor menjadi [1, Tinggi, Lebar, Channel]

            print(f"--> Gambar diproses menjadi tensor untuk model dengan shape: {input_tensor.shape}")

            # --- Langkah 4.4: Menjalankan Inferensi (Prediksi) ---
            print("\n--> Menjalankan inferensi (prediksi deteksi objek)...")
            # Jalankan model pada input tensor. Model akan memprediksi box, kelas, dan skor.
            detections = model(input_tensor)
            print("--> Inferensi selesai.")

            # --- Langkah 4.5: Memproses Hasil Deteksi ---
            # Ambil data deteksi yang valid dari output tensor dan ubah ke numpy array.
            # Output model TF Hub: 'detection_boxes', 'detection_classes', 'detection_scores', 'num_detections'.
            # Kita ambil elemen pertama [0] karena batch size = 1.
            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(np.uint32)
            scores = detections['detection_scores'][0].numpy()
            num_detections = int(detections['num_detections'][0].numpy()) # Jumlah deteksi valid

            # --- Mapping ID Kelas COCO ke Nama Kelas ---
            # Model kita dilatih di dataset COCO. Jadi, ID kelas outputnya merujuk ke kelas-kelas di dataset COCO.
            # Dataset deteksi Anda berisi kendaraan. Karena COCO juga punya kelas kendaraan,
            # model ini bisa mendeteksi kendaraan di gambar Anda.
            # Dictionary ini memetakan ID kelas COCO (dari output model) ke nama kelas (sesuai dokumentasi COCO).
            # Ini dibutuhkan untuk menampilkan nama kelas yang terdeteksi.
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
                # ID kelas kendaraan utama di COCO yang relevan dengan dataset Anda:
                # 3: 'Car', 4: 'Motorcycle', 6: 'Bus', 8: 'Truck', 5: 'Airplane', 9: 'Boat'
                # (Ambulance tidak ada sebagai kelas terpisah di COCO standar, biasanya masuk 'car' atau 'truck')
            }

            # --- Filtering Deteksi ---
            # Hanya tampilkan deteksi yang memiliki skor keyakinan (confidence score) di atas nilai ambang batas (threshold) tertentu.
            # Ini untuk menghindari menampilkan terlalu banyak deteksi palsu dengan skor rendah.
            min_confidence_threshold = 0.5 # Ambil deteksi yang minimal 50% yakin

            filtered_boxes = []
            filtered_classes_ids = []
            filtered_scores = []
            filtered_classes_names = [] # Nama kelas (dari mapping COCO)

            # Loop melalui semua deteksi yang valid (sebanyak num_detections)
            for i in range(num_detections):
                # Jika skor deteksi saat ini di atas nilai threshold yang ditentukan...
                if scores[i] > min_confidence_threshold:
                    class_id = classes[i]
                    # Ambil nama kelas dari mapping COCO. Jika ID tidak ada di mapping, gunakan format 'ID: [number]'.
                    class_name = category_map_coco.get(class_id, f'ID: {class_id}')

                    # (Opsional) Jika hanya ingin menampilkan DETEKSI KENDARAAN saja (Car, Motorcycle, Bus, Truck, Airplane, Boat):
                    # if class_id in [3, 4, 6, 8, 5, 9]:
                    filtered_boxes.append(boxes[i])
                    filtered_classes_ids.append(class_id)
                    filtered_scores.append(scores[i])
                    filtered_classes_names.append(class_name)


            print(f"--> Ditemukan {len(filtered_boxes)} objek valid (skor > {min_confidence_threshold:.2f}) dari kelas COCO yang terdeteksi:")
            for i in range(len(filtered_boxes)):
                 # Tampilkan informasi deteksi yang difilter ke konsol/terminal
                 print(f"  - Objek: {filtered_classes_names[i]} (ID COCO: {filtered_classes_ids[i]}), Skor: {filtered_scores[i]:.2f}")

            # --- Langkah 4.6: Visualisasi Hasil Deteksi ---
            # Gambar bounding box dan label pada gambar asli.
            # Kita menggunakan fungsi bantuan draw_bounding_box_on_image.
            if len(filtered_boxes) > 0:
                # Copy gambar asli (dalam format RGB) untuk digambar agar tidak merubah array asli
                image_with_detections = np.copy(image_np_rgb)

                # Loop melalui deteksi yang sudah difilter
                for i in range(len(filtered_boxes)):
                    # Ambil data box, nama kelas, dan skor untuk deteksi ini
                    ymin, xmin, ymax, xmax = filtered_boxes[i]
                    class_name = filtered_classes_names[i]
                    score = filtered_scores[i]

                    # Panggil fungsi bantuan untuk menggambar box dan label pada gambar
                    image_with_detections = draw_bounding_box_on_image(
                        image_with_detections, # Gambar untuk digambar
                        ymin, xmin, ymax, xmax, # Koordinat box relatif
                        class_name, score,      # Label dan skor
                        color=(0, 255, 0),      # Warna bounding box (Hijau dalam format BGR)
                        font_scale=0.6,         # Ukuran font
                        font_thickness=2        # Ketebalan garis/font
                    )

                print("--> Hasil deteksi digambar pada gambar.")
            else:
                 # Jika tidak ada deteksi yang melewati threshold setelah filtering,
                 # tetap tampilkan gambar asli tanpa box.
                 image_with_detections = np.copy(image_np_rgb)
                 print("--> Tidak ada deteksi yang melewati threshold untuk digambar.")

            # --- Langkah 4.7: Menampilkan Hasil dan Pengujian Langsung ---
            # Menampilkan gambar yang sudah digambar hasil deteksi menggunakan Matplotlib.
            # Jendela gambar akan muncul setelah menjalankan script.
            print("\n--> Menampilkan gambar hasil deteksi.")
            print("Tutup jendela gambar Matplotlib untuk mengakhiri skrip.")
            # Matplotlib mengharapkan gambar dalam format RGB, yang sudah kita siapkan (image_with_detections)
            plt.figure(figsize=(12, 12)) # Menentukan ukuran figure (opsional)
            plt.imshow(image_with_detections) # Menampilkan gambar
            plt.title(f"Hasil Deteksi Objek pada Gambar: {image_file_name}") # Judul gambar
            plt.axis('off') # Menyembunyikan sumbu x dan y
            plt.show() # Menampilkan jendela gambar. Kode akan berhenti di sini sampai jendela ditutup.

            # --- Alternatif: Menampilkan menggunakan jendela pop-up OpenCV ---
            # Jika Anda lebih suka jendela pop-up OpenCV (lebih cepat untuk banyak gambar/video),
            # gunakan kode di bawah. Ingat OpenCV default BGR, jadi perlu konversi kembali.
            # images_bgr = cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Hasil Deteksi Objek Kendaraan (OpenCV)", images_bgr)
            # print("\n--> Jendela OpenCV terbuka. Tekan tombol apapun di jendela tersebut untuk menutup.")
            # cv2.waitKey(0) # Menunggu tombol ditekan di jendela OpenCV
            # cv2.destroyAllWindows() # Menutup semua jendela OpenCV


    else: # Jika model gagal dimuat di Langkah 4.2
        print("\nSkrip berakhir karena model Deep Learning gagal dimuat.")

    print("\n--- Skrip Deteksi Objek Kendaraan Selesai ---")