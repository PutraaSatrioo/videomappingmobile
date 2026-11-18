# Model deteksi harimau

`train_model.py` melatih pendeteksi objek YOLO, lalu mengekspor model ONNX untuk
inferensi langsung di browser melalui ONNX Runtime Web. `index.html` memakai
deteksi kelas harimau di area tengah layar sebagai kondisi target tertangkap.

## Siapkan data

Satu foto `Sumatra_Harimau.png` belum cukup untuk melatih model yang andal. Kumpulkan
banyak foto harimau dengan variasi pose, ukuran, pencahayaan, latar, dan jarak kamera.
Simpan sebagian untuk validasi agar bisa mengukur hasil pada foto yang tidak dipakai
saat training. Anotasikan kotak yang mengelilingi harimau dengan format YOLO Detection
(kelas tunggal `harimau_sumatra`, id kelas `0`) menggunakan alat anotasi seperti
Label Studio atau CVAT, lalu ekspor dataset sebagai YOLO Detection.

Struktur folder yang dibutuhkan:

```text
track/
  dataset/
    images/
      train/  # foto training
      val/    # foto validasi
    labels/
      train/  # .txt pasangan dengan nama file foto
      val/    # .txt pasangan dengan nama file foto
```

Setiap baris file label berisi `0 x_tengah y_tengah lebar tinggi`; semua koordinat
dinormalisasi terhadap ukuran gambar (0 sampai 1). Foto tanpa objek boleh memiliki
file label kosong.

## Latih

Dari folder proyek:

```bash
python -m pip install -r track/requirements.txt
python track/train_model.py
```

Opsi yang tersedia: `--epochs`, `--imgsz`, `--batch`, `--weights`, dan `--device`.
Contoh penggunaan GPU NVIDIA pertama: `python track/train_model.py --device 0`.
Hasil training tersimpan di `track/runs/harimau_sumatra/weights/best.pt`; model
ONNX yang dipakai halaman disalin ke `track/harimau_sumatra.onnx`.

Jalankan halaman melalui server lokal atau HTTPS (jangan buka langsung sebagai
`file://`) agar browser dapat mengambil file model. ONNX Runtime Web dan Supabase
memerlukan koneksi internet. Inferensi berjalan di perangkat pengguna; video kamera
tidak dikirim ke server untuk deteksi. Untuk mengenali hewan di tempat lain dalam
daftar, tambahkan kelas dan data berlabel lalu sesuaikan pemetaan kelas di HTML.
