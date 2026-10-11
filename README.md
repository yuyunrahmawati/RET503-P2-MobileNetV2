# P2 — Klasifikasi Objek Box Industri pada Robot Visi (MobileNetV2)

Project ini mengimplementasikan dan membandingkan tiga strategi pelatihan arsitektur **MobileNetV2** untuk klasifikasi dua kelas komponen objek industri pada sistem robot visi: **Box Merah** (`box_merah`) dan **Box Cokelat** (`box_cokelat`).

---

## Dataset

Dataset dikumpulkan langsung dari kamera robot visi dengan total 200 citra berkualitas tinggi:

| Kelas | Jumlah Citra |
| :--- | :---: |
| Box Merah (`box_merah`) | 100 |
| Box Cokelat (`box_cokelat`) | 100 |
| **Total** | **200** |

Dataset mentah tersimpan dalam folder `box_merah/` dan `box_cokelat/`. Manifest dataset dicatat di `metadata.csv` (menyimpan `path`, `label`, `frame_index`, `group`, `width`, `height`, dan `split`). Pembagian data dijalankan melalui `split.py` dengan metode *chronological temporal split* plus *guard band* (gap) untuk mencegah *data leakage* antar-frame video yang berdekatan secara temporal:

- **Data Training:** 78 citra per kelas (Total: 156 citra)
- **Data Validation:** 18 citra per kelas (Total: 36 citra)
- **Excluded (Guard Band):** 4 citra per kelas (Total: 8 citra)

---

## Rancangan Sistem (CDIO Stage #2 Design)

1. **Misi Proyek:** Sistem robot visi bertugas mengenali dan memilah jenis boks industri secara *real-time* untuk memandu lengan robot manipulator (*sorting arm*).
2. **Kamera & Posisi:** Kamera terpasang pada posisi *top-down / angled* dengan jarak kerja 30–80 cm di atas konveyor/meja kerja. Resolusi citra akuisisi adalah $1280 \times 720$ (box_merah) dan $832 \times 464$ (box_cokelat).
3. **Unit Komputasi:** Komputasi onboard / laptop kontrol industri.
4. **Target Kinerja & Latensi:** 
   - Target akurasi validasi $\ge 95\%$.
   - Kecepatan inferensi onboard $\ge 15\text{ FPS}$ (latensi budget $\le 67\text{ ms}$).
   - Model yang ringan (*lightweight*) untuk efisiensi memori dan bandwidth transfer.

---

## Arsitektur MobileNetV2 & Metode Pelatihan

MobileNetV2 dipilih karena menggunakan arsitektur mutakhir berbasis **Depthwise Separable Convolution** dan **Inverted Residual Blocks with Linear Bottleneck**, menghasilkan jumlah parameter yang jauh lebih kecil (~3,5 juta parameter, ukuran file checkpoint hanya **~9,15 MB**) dibandingkan ResNet-18 (~11,7 juta parameter, ~44,8 MB).

Pelatihan dijalankan melalui skrip `train.py` pada PyTorch dengan:
- Input resolusi $224 \times 224$ piksel.
- Normalisasi standar ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Data augmentasi pada training set: `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`.
- Batch size: 16, Seed: 42, Epochs: 10, Optimizer: Adam dengan Cosine Annealing Learning Rate Scheduler.

### 3 Mode Pelatihan yang Dibandingkan:
1. **Feature Extraction** — Seluruh backbone fitur MobileNetV2 (`model.features`) dibekukan (`requires_grad = False`). Hanya layer `model.classifier` akhir yang dilatih dengan *learning rate* $1\times 10^{-3}$.
2. **Partial Fine-Tuning** — Layer awal dibekukan, sedangkan 2 blok terakhir dari ekstraktor fitur (`model.features[-2:]`: Inverted Residual Block 17 dan Conv2d 1x1 Block 18) bersama `model.classifier` dibuka untuk pelatihan menggunakan *discriminative learning rate* ($1\times 10^{-4}$ untuk blok fitur akhir, dan $1\times 10^{-3}$ untuk classifier).
3. **Training from Scratch** — Seluruh layer MobileNetV2 dilatih dari nol dengan inisialisasi bobot acak tanpa bobot pretrained ImageNet.

---

## Hasil Eksperimen

Berikut adalah hasil empiris pelatihan 10 epoch untuk ketiga mode MobileNetV2:

| Mode Pelatihan | Validation Accuracy Terbaik | Epoch Terbaik | Epoch Capai $\ge 90\%$ | Waktu Training (CPU) | Ukuran Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Partial Fine-Tuning** | **100,00%** | **1** | **1** | **~72,1 detik** | **9,15 MB** |
| **Feature Extraction** | **100,00%** | **1** | **1** | **~125,7 detik** | **9,15 MB** |
| **Training from Scratch** | **100,00%** | **8** | **8** | **~151,5 detik** | **9,15 MB** |

### Hasil Pengukuran Latensi Inferensi (Batch Size = 1)

| Checkpoint Model | Device | Latensi Median | Latensi Rata-rata | Latensi P95 | Estimasi Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mobilenet_v2_partial.pth` | CPU | **18,77 ms** | 21,28 ms | 38,86 ms | **~53,2 FPS** |
| `mobilenet_v2_scratch.pth` | CPU | **21,17 ms** | 23,49 ms | 41,87 ms | **~47,2 FPS** |
| `mobilenet_v2_feature.pth` | CPU | **24,19 ms** | 27,41 ms | 51,69 ms | **~41,3 FPS** |

Model terpilih untuk implementasi adalah **Partial Fine-Tuning (`mobilenet_v2_partial.pth`)** karena:
1. Meraih akurasi validasi optimal ($100,00\%$) sejak **epoch pertama**.
2. Memiliki waktu konvergensi dan durasi pelatihan paling cepat (**72,1 detik**).
3. Menghasilkan latensi inferensi terbaik (**18,77 ms per frame**, setara **~53,2 FPS** pada CPU), jauh melampaui target latensi minimum robot ($67\text{ ms}$ / $15\text{ FPS}$).

Artefak hasil eksperimen:
- `results/experiments.csv` — Ringkasan metrik akurasi, epoch konvergensi, dan waktu training.
- `results/latency.csv` — Catatan detail latensi inferensi (mean, median, p95).
- `results/accuracy_comparison.png` — Grafik perbandingan kurva akurasi validasi antar-mode per epoch.
- `results/feature_history.csv` — Log riwayat training per epoch mode Feature Extraction.
- `results/partial_history.csv` — Log riwayat training per epoch mode Partial Fine-Tuning.
- `results/scratch_history.csv` — Log riwayat training per epoch mode Scratch.
- `models/mobilenet_v2_partial.pth` — Bobot model terbaik siap pakai.

---

## Evaluasi dan Analisis Keterbatasan

1. **Efisiensi Arsitektur MobileNetV2:** Berkat penggunaan *Inverted Residuals* dan *Depthwise Separable Convolutions*, bobot model terkompresi hingga **9,15 MB** (hampir 5 kali lipat lebih ringan dibanding ResNet-18 yang berukuran 44,8 MB), menjadikannya sangat ideal untuk perangkat tertanam (*edge device* seperti Raspberry Pi atau robot industri berdaya rendah).
2. **Keunggulan Transfer Learning vs Scratch:** Transfer learning menunjukkan keunggulan mutlak. Model Partial Fine-Tuning dan Feature Extraction langsung mencapai 100% pada epoch 1. Sebaliknya, Training from Scratch membutuhkan 8 epoch untuk keluar dari tebakan acak ($50,00\%$) karena arsitektur depthwise separable convolution membutuhkan representasi fitur bertingkat yang sulit dipelajari dari dataset kecil (156 citra latih) tanpa bobot awal ImageNet.
3. **Pencegahan Data Leakage:** Skema pemisahan data temporal dengan *guard band* pada `split.py` berhasil menjamin bahwa frame validasi tidak memiliki kesamaan temporal yang berlebihan dengan frame latih.
4. **Analisis Latensi:** Latensi inferensi sebesar **18,77 ms** pada CPU membuktikan bahwa MobileNetV2 mampu beroperasi pada kecepatan lebih dari **50 FPS**, sehingga sangat responsif untuk sistem penyortiran boks industri *real-time*.

---

## Panduan Menjalankan Kode (Step-by-Step)

### Langkah 1: Persiapan Environment & Dependensi
Pastikan Python 3.10+ atau uv terpasang. Buat virtual environment dan instal pustaka yang diperlukan:

```bash
# Menggunakan virtualenv / uv
uv venv .venv
uv pip install torch torchvision pillow matplotlib
```

Aktifkan environment:
- **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```

---

### Langkah 2: Pemisahan Dataset (`split.py`)
Membagi data di `metadata.csv` menjadi subset training dan validation secara sekuensial temporal dengan *guard band*:

```bash
python split.py --val-fraction 0.2 --gap 2
```

Output:
```text
box_cokelat: train=78 validation=18 excluded=4
box_merah: train=78 validation=18 excluded=4
```

---

### Langkah 3: Pelatihan Model 3 Mode MobileNetV2 (`train.py`)
Melatih 3 mode eksperimen MobileNetV2 secara otomatis (feature, partial, scratch) masing-masing 10 epoch:

```bash
python train.py --epochs 10 --batch-size 16 --seed 42
```

Output yang dihasilkan:
- Checkpoint: `models/mobilenet_v2_feature.pth`, `models/mobilenet_v2_partial.pth`, `models/mobilenet_v2_scratch.pth`
- Riwayat log: `results/*_history.csv` & `results/experiments.csv`
- Visualisasi: `results/accuracy_comparison.png`

---

### Langkah 4: Pengukuran Kecepatan Inferensi (`latency.py`)
Mengukur latensi waktu inferensi 1 citra (batch size = 1) sebanyak 100 iterasi dengan 20 iterasi warmup:

```bash
python latency.py --checkpoint models/mobilenet_v2_partial.pth --image box_merah/box_merah__frame_000000000.jpg
python latency.py --checkpoint models/mobilenet_v2_feature.pth --image box_merah/box_merah__frame_000000000.jpg
python latency.py --checkpoint models/mobilenet_v2_scratch.pth --image box_merah/box_merah__frame_000000000.jpg
```

---

### Langkah 5: Uji Prediksi Foto Baru (`predict.py`)
Melakukan inferensi klasifikasi citra secara langsung menggunakan model MobileNetV2 terbaik:

```bash
python predict.py box_cokelat/box_cokelat__frame_000000000.jpg
python predict.py box_merah/box_merah__frame_000000000.jpg
```

Contoh Output Prediksi:
```text
========================================
 File Gambar    : box_cokelat__frame_000000000.jpg
 Hasil Prediksi : BOX_COKELAT
 Tingkat Keyakinan (Confidence): 75.51%
========================================
========================================
 File Gambar    : box_merah__frame_000000000.jpg
 Hasil Prediksi : BOX_MERAH
 Tingkat Keyakinan (Confidence): 77.03%
========================================
```
