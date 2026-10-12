# 🌾 NPK Phonska IV — Forecasting Intelligence (Streamlit)

Dashboard interaktif untuk membandingkan 9 metode prediksi multivariat
(VAR, SVR, Random Forest, XGBoost, LightGBM, LSTM, RNN, GRU, TCN) beserta
control chart residual MEWMA & Adaptive-DHWMA.

## Struktur repo

```
.
├── app.py                # antarmuka dashboard (Streamlit + Plotly)
├── engine.py             # komputasi: 9 metode, control chart, ekspor Excel
├── requirements.txt
├── .streamlit/config.toml
└── data/
    └── NPK_Phonska_IV_Imputasi_ML.xlsx   # <-- taruh dataset di sini
```

Kolom dataset: `Time`, `H2O`, `N`, `P2O5`, `K2O` (interval 4 jam, sudah tanpa missing value).
Aplikasi mencari dataset otomatis di `data/` (lihat `CONFIG['DATA_CANDIDATES']` di `app.py`);
bila tidak ditemukan, pengguna bisa mengunggah file lewat sidebar.

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy gratis (Streamlit Community Cloud)

1. Push seluruh folder ini (termasuk `data/`) ke GitHub.
2. Buka share.streamlit.io → *New app* → pilih repo, branch, dan file `app.py`.
3. Pilih Python 3.11 atau 3.12 pada *Advanced settings*.

> Memori gratis terbatas (~1 GB). TensorFlow cukup berat; jika app crash, hapus baris
> `tensorflow-cpu` dari `requirements.txt` (metode LSTM/RNN/GRU/TCN otomatis dilewati)
> atau gunakan preset **Cepat (demo)**.

## Konfigurasi

Semua parameter dapat diubah dari sidebar (form → tombol **Jalankan Analisis**):
variabel, sliding window, rasio train/test, seed, kriteria pemilihan window, metode yang dijalankan,
grid SVR, iterasi Random Forest, grid & epoch deep learning, parameter control chart (α, λ, γ, Monte Carlo).
Nilai default berada di dictionary `CONFIG` dan `PRESETS` pada bagian atas `app.py`.

Hasil dengan konfigurasi + data yang sama disimpan di folder `.cache_hasil/` sehingga
tidak perlu dihitung ulang (folder ini tidak perlu di-commit; tambahkan ke `.gitignore`).

## Isi dashboard

| Tab | Isi |
|---|---|
| Beranda | KPI, alur analisis, konfigurasi aktif |
| Eksplorasi Data | animasi deret waktu, z-score + range slider, korelasi, ACF, peta panas harian×jam, violin latih/uji, uji ADF |
| Perbandingan Metode | papan peringkat, balapan metode per window (animasi), bubble animasi, heatmap, radar, bump chart, treemap, efek window, waktu |
| Prediksi & Residual | animasi aktual vs prediksi, RMSE kumulatif animasi, scatter, violin residual, heatmap galat |
| Control Chart | kalibrasi UCL, chart animasi MEWMA & Adaptive-DHWMA, OOC per metode, barcode sinyal |
| Metode & Dampak | penjelasan tiap metode, hyperparameter, metrik, dampak pada dataset |
| Ekspor | Excel lengkap, CSV prediksi, JSON konfigurasi |

Setiap grafik/tabel diikuti kotak **💡 Interpretasi** yang dihitung otomatis dari hasil.
