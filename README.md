# Tugas Individu: Eksplorasi Autoencoder

## Deskripsi
Proyek ini membangun dan mengeksplorasi **Autoencoder** dengan dataset **FashionMNIST**.  
Model yang diuji mencakup:
- **Convolutional Autoencoder (AE)**: encoder–decoder sederhana untuk rekonstruksi.
- **Variational Autoencoder (VAE)**: menambahkan regularisasi probabilistik dengan KL-divergence.

Tujuan utama bukan hanya menghasilkan rekonstruksi yang baik, tetapi juga memahami bagaimana struktur encoder–decoder memengaruhi representasi laten dan kualitas rekonstruksi.

---

## Arsitektur
- **Encoder**: 3 lapis Conv2d + BatchNorm + ReLU → Fully Connected → Latent space
- **Decoder**: 3 lapis ConvTranspose2d + BatchNorm + ReLU → Tanh output
- **Latent Dimensi**: default `2` (mudah divisualisasi, bisa diubah)
- **Loss**:
  - AE: Mean Squared Error (MSE)
  - VAE: MSE + KL-divergence

---

## Hasil Ringkas
- **Rekonstruksi**: Gambar hasil Autoencoder masih mempertahankan bentuk utama objek FashionMNIST, meski detail halus sedikit hilang. VAE menghasilkan variasi lebih halus tetapi kadang blur.
- **Representasi Laten**:
  - Dengan `latent_dim=2`, data membentuk kluster yang sesuai label walau tidak dilatih secara supervised.
  - Menggunakan **t-SNE** (untuk dimensi >2) juga menunjukkan pola pemisahan antar kelas.
- **Interpolasi Laten**: Transisi antar gambar berjalan mulus, menunjukkan latent space berhasil menangkap semantik data.

---

## Petunjuk Eksekusi
1. **Clone / Unduh repository**
   ```bash
   git clone <url-repo>
   cd <nama-folder>
   
2. Install dependencies
pip install torch torchvision matplotlib scikit-learn

3. Buka Jupyter Notebook
jupyter notebook

Jalankan notebook:

- FashionMNIST_AE_VAE_Fixed.ipynb
atau
- t6-eksplorasi-autoencoder.ipynb

4. Atur konfigurasi di bagian awal notebook:
- CFG.model_type = "AE" atau "VAE"
- CFG.latent_dim = 2 atau lebih (mis. 16, 32)
- CFG.epochs sesuai kebutuhan
  
5. Jalankan semua sel untuk:
- Melatih model
- Melihat rekonstruksi
- Visualisasi ruang laten
- Interpolasi laten
- Menyimpan model (.pt) dan latents (.npy)
