# Model deteksi badak dan harimau

`train_model.py` melatih pendeteksi objek YOLO, lalu mengekspor model ONNX untuk
inferensi langsung di browser melalui ONNX Runtime Web. `index.html` memakai
deteksi kelas harimau di area tengah layar sebagai kondisi target tertangkap.

## Siapkan data

Satu foto `Sumatra_Harimau.png` belum cukup untuk melatih model yang andal. Kumpulkan
banyak foto harimau dengan variasi pose, ukuran, pencahayaan, latar, dan jarak kamera.
Simpan sebagian untuk validasi agar bisa mengukur hasil pada foto yang tidak dipakai
saat training. Anotasikan kotak yang mengelilingi harimau dengan format YOLO Detection
(tiga kelas: `harimau_sumatra` id `0`, `badak_sumatra` id `1`, dan
`merak_jawa` id `2`) menggunakan alat anotasi seperti Label Studio atau CVAT,
lalu ekspor dataset sebagai YOLO Detection.

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
python track/generate_from_single_image.py --train-per-class 100 --val-per-class 25
python track/train_model.py
```

Opsi yang tersedia: `--epochs`, `--imgsz`, `--batch`, `--weights`, dan `--device`.
Contoh penggunaan GPU NVIDIA pertama: `python track/train_model.py --device 0`.
Hasil training tersimpan di `track/runs/badak_harimau/weights/best.pt`; model
ONNX yang dipakai halaman disalin ke `track/badak_harimau.onnx`.

Jalankan halaman melalui server lokal atau HTTPS (jangan buka langsung sebagai
`file://`) agar browser dapat mengambil file model. ONNX Runtime Web dan Supabase
memerlukan koneksi internet. Inferensi berjalan di perangkat pengguna; video kamera
tidak dikirim ke server untuk deteksi. Untuk mengenali hewan di tempat lain dalam
daftar, tambahkan kelas dan data berlabel lalu sesuaikan pemetaan kelas di HTML.


Dataset saat ini tetap perlu anotasi badak pada split train dan val. Skrip training sengaja berhenti jika salah satu kelas belum muncul di split tersebut. Setelah ekspor ONNX tersedia, halaman kamera menampilkan nama kelas dan confidence di atas setiap hewan yang terdeteksi.

Generator membuat 100 gambar training dan 25 validasi per kelas dari PNG transparan,
termasuk `merak_jawa` dengan ID kelas `2`. Variasi sintetis dari satu gambar sumber
berguna untuk alur awal, tetapi model tetap perlu diuji dengan foto nyata yang beragam.
