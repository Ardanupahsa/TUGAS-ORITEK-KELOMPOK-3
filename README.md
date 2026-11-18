# TUGAS ORITEK Computational chemistry: Predict aqueous solubility

KELOMPOK-3

1. Arief Efendi G1025042
2. Arief Hafid Zasella G1A025052
3. Rikza Zalfanara G1A025058
4. Fitria Wulandini G1A025084
5. Vikri Abdullah Razzak G1A025092
6. Ardanu Pahsa Daniswara G1A025106
7. Bonardo Bastian Sihaloho G1A025118
## 1. Deskripsi Proyek

Proyek ini merupakan implementasi kecerdasan buatan (Machine Learning) berbasis Python yang bertujuan untuk memprediksi tingkat kelarutan senyawa kimia dalam air (dalam satuan LogS) berdasarkan karakteristik parameter fisikokimia molekulnya[cite: 1]. Pemodelan dilakukan dengan memanfaatkan algoritma **Random Forest Regressor**, yang digunakan untuk mempelajari pola hubungan antarparameter kimia dan menghasilkan estimasi nilai kelarutan[cite: 1]. Proyek ini menggunakan dataset terkurasi `curated-solubility-dataset.csv` dengan tujuh fitur parameter molekul utama sebagai variabel input[cite: 1]. Data tersebut dibagi menjadi data latih dan data uji sebelum diberikan kepada model untuk proses pelatihan[cite: 1]. Hasil akhir berupa skor evaluasi performa model ($R^2$ Score dan RMSE), grafik visualisasi analisis fitur serta perbandingan hasil tebakan AI, dan program simulasi interaktif untuk memprediksi kelarutan senyawa baru[cite: 1].

## 2. Latar Belakang dan Studi Kasus

Kelarutan suatu senyawa kimia dalam air merupakan salah satu sifat fisikokimia paling fundamental yang sangat menentukan efektivitasnya dalam berbagai bidang, seperti pengembangan formulasi obat-obatan (farmasi), kimia lingkungan, maupun sintesis material baru. Pengukuran tingkat kelarutan secara konvensional di laboratorium sering kali membutuhkan waktu yang lama, biaya eksperimen yang tinggi, serta sumber daya kimia yang tidak sedikit. Melalui pendekatan komputasi dan ketersediaan data karakteristik molekul (seperti berat molekul, koefisien partisi lemak, dan ikatan hidrogen), pola kelarutan berbagai senyawa dapat dianalisis secara matematis. Oleh karena itu, proyek ini menerapkan algoritma Machine Learning berbasis Random Forest untuk melakukan prediksi kelarutan senyawa kimia secara cepat dan terukur berdasarkan profil parameter molekulnya[cite: 1].

## 3. Tujuan Proyek

Tujuan dari proyek ini adalah:

1. Menerapkan algoritma Machine Learning (Random Forest Regressor) dalam bidang kimia komputasi[cite: 1].
2. Memprediksi nilai kelarutan senyawa kimia dalam air (`LogS`) berdasarkan tujuh parameter fisikokimia molekul[cite: 1].
3. Melakukan pembagian dataset menjadi data latih (80%) dan data uji (20%) untuk pengujian model secara objektif[cite: 1].
4. Mengevaluasi performa prediksi model menggunakan metrik evaluasi regresi berupa $R^2$ Score dan Root Mean Squared Error (RMSE)[cite: 1].
5. Menampilkan hasil analisis pengaruh parameter kimia dan perbandingan prediksi melalui grafik visualisasi[cite: 1].
6. Menyediakan program simulasi interaktif berbasis teks untuk menguji kelarutan senyawa baru secara langsung[cite: 1].

## 4. Dataset yang Digunakan

Dataset yang digunakan adalah **Curated Solubility Dataset** yang diperoleh melalui Kaggle. Sumber dataset: https://www.kaggle.com/datasets/sorkun/curated-solubility-dataset

Dataset terdiri dari satu file utama bernama **`curated-solubility-dataset.csv`** yang memuat ribuan data senyawa kimia beserta nilai kelarutannya dalam air (dalam satuan `LogS`). Pembagian dataset untuk pemodelan dilakukan secara acak (`random_state=42`) dengan proporsi:

* Training Set: 80% data senyawa kimia (digunakan untuk melatih model AI mempelajari pola karakteristik molekul).
* Testing Set: 20% data senyawa kimia (digunakan untuk pengujian independen terhadap akurasi prediksi model).

Dalam proses pemodelannya, sistem mengekstrak tujuh parameter fisikokimia molekul sebagai fitur input (`X`), yaitu: **a. `MolWt`** (Berat Molekul). **b. `MolLogP`** (Koefisien partisi oktanol-air / kelarutan lemak). **c. `MolMR`** (Refraktivitas Molar). **d. `HeavyAtomCount`** (Jumlah atom berat selain hidrogen). **e. `NumHAcceptors`** (Jumlah akseptor ikatan hidrogen). **f. `NumHDonors`** (Jumlah donor ikatan hidrogen). **g. `NumRotatableBonds`** (Jumlah ikatan yang dapat berputar). Sementara itu, kolom **`Solubility`** berperan sebagai target output (`y`) yang merepresentasikan nilai kelarutan asli setiap senyawa.

## 5. Model Machine Learning yang Digunakan

Model yang digunakan dalam proyek ini adalah **Random Forest Regressor** yang diimplementasikan melalui pustaka **Scikit-learn** (`sklearn.ensemble.RandomForestRegressor`).

Random Forest Regressor merupakan algoritma *ensemble learning* berbasis pohon keputusan (*Decision Trees*) yang dirancang untuk menangani permasalahan regresi numerik. Model ini bekerja dengan membangun **100 pohon keputusan (`n_estimators=100`)** secara paralel pada berbagai sub-sampel data, kemudian merata-ratakan hasil prediksi dari seluruh pohon tersebut untuk menghasilkan nilai estimasi `LogS` yang akurat serta meminimalkan risiko *overfitting*. Model dilatih secara langsung dari awal (*from scratch*) menggunakan data latih, dan dievaluasi menggunakan metrik $R^2$ Score (mencapai **0.75**) serta Root Mean Squared Error / RMSE (sebesar **1.17**).

## 6. Lingkungan dan Teknologi

Proyek dikembangkan menggunakan lingkungan komputasi berbasis Python (Jupyter Notebook / Google Colab). Spesifikasi dan teknologi yang digunakan meliputi:

* Platform: Jupyter Notebook / Google Colab / VS Code.
* Bahasa pemrograman: Python 3.
* Model: Random Forest Regressor (`n_estimators=100`, `random_state=42`).
* Machine Learning Library: Scikit-learn (`train_test_split`, `RandomForestRegressor`, `r2_score`, `mean_squared_error`).
* Data Processing: Pandas dan NumPy.
* Visualisasi: Matplotlib dan Seaborn.
* Antarmuka Prediksi: Command Line Interface (CLI) interaktif berbasis fungsi `input()` Python.

## 7. Tahapan Implementasi

Implementasi proyek dibagi menjadi beberapa bagian utama berdasarkan urutan eksekusi kode dari awal hingga akhir di dalam notebook.

### Part 1 – Library Import
Mengimpor seluruh pustaka Python yang dibutuhkan pada awal sel pertama, meliputi `pandas` dan `numpy` untuk pengolahan data, `matplotlib.pyplot` serta `seaborn` untuk visualisasi grafik, dan modul dari `sklearn` (`train_test_split`, `RandomForestRegressor`, `r2_score`, `mean_squared_error`) untuk pemodelan serta evaluasi kecerdasan buatan.

### Part 2 – Data Loading & Feature Selection
Membaca file dataset `curated-solubility-dataset.csv` menggunakan `pd.read_csv()`. Selanjutnya, sistem memilih tujuh kolom parameter fisikokimia molekul sebagai variabel input (`X`), yaitu `MolWt`, `MolLogP`, `MolMR`, `HeavyAtomCount`, `NumHAcceptors`, `NumHDonors`, dan `NumRotatableBonds`, serta menetapkan kolom `Solubility` sebagai variabel target (`y`).

### Part 3 – Dataset Splitting (Train-Test Split)
Membagi dataset ke dalam dua bagian secara acak menggunakan fungsi `train_test_split` dengan proporsi 80% untuk data latih (`X_train`, `y_train`) dan 20% untuk data uji (`X_test`, `y_test`) menggunakan parameter `random_state=42` agar pembagian data bersifat konsisten saat kode dijalankan ulang.

### Part 4 – Model Initialization & Training
Menginisialisasi algoritma **Random Forest Regressor** (`n_estimators=100`, `random_state=42`) dan melatihnya menggunakan data latih melalui fungsi `model.fit(X_train, y_train)` agar model mempelajari pola hubungan antara karakteristik molekul dan nilai kelarutannya.

### Part 5 – Model Prediction & Evaluation
Melakukan pengujian terhadap data uji menggunakan fungsi `model.predict(X_test)`. Hasil prediksi (`y_pred`) kemudian dibandingkan dengan nilai kelarutan asli (`y_test`) untuk menghitung skor koefisien determinasi ($R^2$ Score = **0.75**) dan tingkat kesalahan Root Mean Squared Error (RMSE = **1.17**), lalu menampilkannya ke layar.

### Part 6 – Results Visualization
Membuat figur visualisasi dengan ukuran `14x5` menggunakan Matplotlib yang terbagi menjadi dua subplot berdampingan:
* **Grafik 1 (Feature Importance):** Grafik batang horizontal (`barh`) yang menampilkan tingkat pengaruh masing-masing dari ketujuh parameter kimia terhadap prediksi kelarutan.
* **Grafik 2 (Actual vs Predicted):** Grafik *scatter plot* yang membandingkan nilai `Solubility` aktual dengan hasil tebakan model AI, dilengkapi garis diagonal merah putus-putus sebagai acuan prediksi ideal.

### Part 7 – Interactive User Input & New Molecule Prediction
Menjalankan sel kedua yang berfungsi sebagai antarmuka simulasi interaktif berbasis teks (CLI). Program meminta pengguna memasukkan tujuh nilai numerik parameter molekul baru secara berurutan melalui fungsi `float(input())`, menyusunnya ke dalam bentuk `DataFrame`, dan memprediksi estimasi nilai `LogS` menggunakan model yang telah dilatih pada tahap sebelumnya.

### Part 8 – Solubility Classification Output
Menerjemahkan angka hasil prediksi `LogS` ke dalam kategori tingkat kelarutan yang mudah dipahami menggunakan percabangan logika (`if-elif-else`), yakni **Sangat Mudah Larut dalam Air** (`LogS > -2`), **Cukup Larut dalam Air** (`-4 < LogS <= -2`), atau **Sukar / Tidak Larut dalam Air** (`LogS <= -4`), kemudian menampilkan hasil akhirnya ke pengguna.
```[cite: 1]

## 8. Metode dan Teknik yang Digunakan

Beberapa metode utama yang diterapkan dalam proyek ini adalah: **Feature Selection** Digunakan untuk mengekstrak tujuh parameter fisikokimia molekul yang paling relevan (`MolWt`, `MolLogP`, `MolMR`, `HeavyAtomCount`, `NumHAcceptors`, `NumHDonors`, dan `NumRotatableBonds`) sebagai variabel prediktor. **Train-Test Split** Membagi dataset secara acak menggunakan parameter `test_size=0.2` dan `random_state=42` untuk memisahkan data latih (80%) dan data uji (20%) guna menjaga objektivitas pengujian. **Ensemble Learning (Random Forest)** Membangun 100 pohon keputusan (`n_estimators=100`) secara paralel dan menggabungkan rata-rata hasil prediksinya untuk menghasilkan estimasi nilai numerik `LogS` yang stabil serta mencegah *overfitting*. **Feature Importance Analysis** Mengekstrak bobot kontribusi dari setiap fitur molekul untuk mengidentifikasi parameter kimia yang paling dominan dalam menentukan kelarutan. **Threshold-Based Classification** Mengonversi hasil prediksi regresi numerik `LogS` ke dalam tiga kategori kelarutan berbasis ambang batas nilai (-2 dan -4) pada program simulasi interaktif.

## 9. Output yang Dihasilkan

Program menghasilkan beberapa output utama sebagai berikut: **a. Skor Evaluasi Teks** Berupa ringkasan performa model pada konsol yang menampilkan nilai koefisien determinasi ($R^2$ Score = 0.75) dan Root Mean Squared Error (RMSE = 1.17). **b. Visualisasi Analisis Model** Menampilkan figur berisi dua grafik berdampingan, yaitu:

1. Feature Importance Bar Chart (grafik batang horizontal yang mengurutkan pengaruh ketujuh parameter fisikokimia molekul).
2. Actual vs Predicted Scatter Plot (grafik sebar yang membandingkan nilai kelarutan aktual dengan hasil prediksi AI beserta garis diagonal acuan ideal).

**c. Hasil Prediksi Simulasi Interaktif** Menampilkan estimasi nilai `LogS` senyawa baru beserta label kategori kelarutannya (*Sangat Mudah Larut dalam Air*, *Cukup Larut dalam Air*, atau *Sukar / Tidak Larut dalam Air*) berdasarkan tujuh parameter molekul yang diinputkan pengguna secara langsung.

## 10. Evaluasi Model

Evaluasi dilakukan untuk mengetahui tingkat akurasi dan presisi model regresi dalam memprediksi nilai kelarutan senyawa kimia (`LogS`). Metrik evaluasi yang digunakan meliputi: **R² Score (Coefficient of Determination)** Mengukur seberapa besar proporsi variansi dari nilai kelarutan aktual yang dapat dijelaskan oleh model berdasarkan tujuh fitur molekul, di mana proyek ini mencapai skor **0.75** (75%). **RMSE (Root Mean Squared Error)** Mengukur rata-rata tingkat kesalahan prediksi model dalam satuan yang sama dengan target (`LogS`), dengan hasil sebesar **1.17** yang menunjukkan penyimpangan prediksi yang relatif rendah pada rentang skala kelarutan senyawa. Evaluasi utama dilakukan menggunakan *testing set* (20% dari total dataset) yang belum pernah dilihat oleh model selama proses pelatihan.
