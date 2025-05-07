Aplikasi Web Deteksi Objek Kendaraan Sederhana
Deskripsi Proyek
Proyek ini adalah implementasi sederhana dari aplikasi web untuk mendeteksi objek kendaraan dalam gambar menggunakan model Deep Learning pre-trained dari TensorFlow Hub dan framework web Flask di Python. Pengguna dapat mengunggah gambar melalui antarmuka web, dan aplikasi akan menampilkan gambar hasil deteksi dengan bounding box serta daftar objek yang terdeteksi.

Fitur Utama
- Mengunggah gambar melalui antarmuka web.
- Menjalankan inferensi deteksi objek menggunakan model Deep Learning pre-trained.
- Menampilkan gambar hasil deteksi dengan bounding box dan label.
- Menyajikan daftar objek yang terdeteksi beserta skor keyakinan.

Prasyarat
- Python 3.6+: Anda bisa mengunduhnya dari python.org.
- pip: Biasanya sudah terinstal bersama Python. Virtual Environment (opsional, tapi sangat direkomendasikan): Seperti venv (bawaan Python) atau conda.

Instalasi
Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek ini di lingkungan lokal Anda:

pip install Flask tensorflow tensorflow-hub opencv-python matplotlib numpy

Struktur File
Struktur proyek diharapkan seperti ini:

.
├── app.py              # Kode backend Flask
├── templates/          # Folder untuk template HTML
│   └── index.html      # Template frontend
└── static/             # Folder untuk menyimpan gambar hasil (akan dibuat otomatis)

Pastikan file index.html berada di dalam folder templates dan file app.py berada di direktori utama proyek. Folder static akan dibuat secara otomatis saat aplikasi dijalankan.

Penggunaan
Jalankan Aplikasi Flask:
Buka terminal atau Command Prompt, navigasikan ke direktori utama proyek Anda (tempat app.py berada), pastikan virtual environment aktif, lalu jalankan perintah:

python app.py

Anda akan melihat output di terminal yang menunjukkan bahwa server Flask telah berjalan dan alamat yang bisa diakses (biasanya http://127.0.0.1:5000/).

Akses Aplikasi:
Buka browser web Anda dan kunjungi alamat yang tertera di output terminal (misalnya, http://127.0.0.1:5000/).

Unggah Gambar:
Di halaman web yang terbuka, klik tombol untuk memilih file, pilih gambar yang berisi kendaraan dari komputer Anda, lalu klik tombol "Deteksi Objek".

Lihat Hasil:
Aplikasi akan memproses gambar, menjalankan deteksi, dan menampilkan gambar hasil dengan bounding box serta daftar objek yang terdeteksi di halaman yang sama.

Model Deep Learning
Aplikasi ini menggunakan model deteksi objek SSD MobileNet V2 FPNLite dari TensorFlow Hub. Model ini adalah model pre-trained yang dilatih pada dataset besar seperti COCO, yang mencakup berbagai kategori objek termasuk kendaraan. Penggunaan model pre-trained memungkinkan kita untuk langsung melakukan inferensi tanpa perlu melatih model dari awal.

Dataset (Catatan)
Meskipun aplikasi ini berfokus pada inferensi menggunakan model pre-trained, dataset teranotasi memainkan peran vital dalam pengembangan model Computer Vision. Contoh dataset yang relevan adalah "Vehicle Detection Image Dataset" yang tersedia di Kaggle (https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset). Dataset semacam ini berisi gambar dengan anotasi bounding box dan label yang akurat, yang sangat berguna untuk:
- Melatih model deteksi objek kustom.
- Melakukan fine-tuning pada model pre-trained agar lebih spesifik pada jenis data tertentu.
- Mengevaluasi kinerja model secara kuantitatif.

Potensi Pengembangan Lebih Lanjut
Proyek dasar ini dapat dikembangkan lebih lanjut dengan menambahkan fitur-fitur seperti:
- Deteksi pada video atau live stream dari webcam.
- Menambahkan lebih banyak kategori objek untuk dideteksi.
- Menggunakan model deteksi objek yang berbeda atau melatih model kustom.
- Meningkatkan antarmuka pengguna (UI) dan pengalaman pengguna (UX).
