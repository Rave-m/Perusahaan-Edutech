# Proyek Data Science: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan Institusi pendidikan tinggi yang sedang menghadapi tantangan signifikan terkait tingkat putus sekolah (dropout), mendaftar dan keberhasilan akademik mahasiswa. Tingginya angka dropout tidak hanya berdampak pada reputasi institusi tetapi juga pada sumber daya yang diinvestasikan dalam pendidikan mahasiswa. Dengan memahami faktor-faktor yang mempengaruhi keputusan mahasiswa untuk tetap melanjutkan atau menghentikan studi mereka, institusi dapat mengimplementasikan strategi yang lebih efektif untuk meningkatkan retensi dan keberhasilan akademik.

### Permasalahan Bisnis:

Meskipun institusi pendidikan tinggi menyediakan berbagai program studi, terdapat variasi signifikan dalam tingkat retensi dan keberhasilan akademik mahasiswa di berbagai program tersebut. Tingginya tingkat putus sekolah dan rendahnya kinerja akademik di beberapa program studi mengindikasikan adanya faktor-faktor yang belum sepenuhnya dipahami atau ditangani. Oleh karena itu, diperlukan model klasifikasi yang dapat memprediksi tingkat putus sekolah dan keberhasilan akademik mahasiswa berdasarkan data pendaftaran dan kinerja akademik awal mereka. Berdasarkan hal tersebut berikut ini adalah pertanyaan bisnis yang dapat dibuat:

1. Faktor apa saja yang paling signifikan mempengaruhi keputusan mahasiswa untuk dropout di institusi ini?
2. Seberapa akurat model klasifikasi dalam memprediksi dropout dan keberhasilan akademik mahasiswa?
3. Langkah-langkah apa yang dapat diambil oleh pihak institusi untuk menurunkan tingkat dropout berdasarkan prediksi model?

### Cakupan Proyek:

- Analisis dan eksplorasi dataset mahasiswa.
- Pembersihan dan persiapan data.
- Pembangunan dan evaluasi beberapa model machine learning.
- Visualisasi hasil menggunakan Looker Studio.
- Pengembangan aplikasi dashboard dan prediksi berbasis Streamlit.

### Persiapan

#### Dataset:

Sumber Data berasal dari github dengan link berikut: https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

Detail kolom dataset:

- **Marital status**: The marital status of the student. -**Application mode**: The method of application used by the student.
- **Application order**: The order in which the student applied.
- **Course**: The course taken by the student.
- **Daytime/evening attendance**: Whether the student attends classes during the day or in the evening.
- **Previous qualification**: The qualification obtained by the student before enrolling in higher education.
- **Previous qualification (grade)**: Grade of previous qualification (between 0 and 200)
- **Nacionality**: The nationality of the student.
- **Mother's qualification**: The qualification of the student's mother.
- **Father's qualification**: The qualification of the student's father.
- **Mother's occupation**: The occupation of the student's mother.
- **Father's occupation**: The occupation of the student's father.
- **Admission grade**: Admission grade (between 0 and 200)
- **Displaced**: Whether the student is a displaced person.
- **Educational special needs**: Whether the student has any special educational needs.
- **Debtor**: Whether the student is a debtor.
- **Gender**: The gender of the student.
- **Scholarship holder**: Whether the student is a scholarship holder.
- **Age at enrollment**: The age of the student at the time of enrollment.
- **International**: Whether the student is an international student.
- **Curricular units 1st sem (credited)**: The number of curricular units credited by the student in the first semester.
- **Curricular units 1st sem (enrolled)**: The number of curricular units enrolled by the student in the first semester.
- **Curricular units 1st sem (evaluations)**: The number of curricular units evaluated by the student in the first semester.
- **Curricular units 1st sem (approved)**: The number of curricular units approved by the student in the first semester.

Setup environment:

1. Membuat virtual environment

```
python -m venv .venv
```

2. Aktivasi environment

```
.venv/scripts/activate
```

3. Menginstall depedensi yang dibutuhkan

```
pip install -r requirements.txt
```

#### Data Preparation

Pada proses persiapan data meliputi beberapa tahap yaitu sebagai berikut:

1. Memeriksa dengan melihat apakah ada anomali atau penyimpangan pada dataset yang dimiliki
2. Memeriksa distribusi kolom target untuk nantinya dilihat apakah akan dilakukan balancing data atau tidak (pada akhirnya dicase ini kita harus melakukan balancing data target karna data target terindikasi imbalanced) .
3. Mengubah kolom target menjadi data numerikal dengan menggunakan `LabelEncoder`.
4. Memilih kolom-kolom atau fitur yang memiliki korelasi yang cukup dengan kolom target.
5. Memisahkan data menjadi 2 bagian yaitu data training dan data testing dengan perbandingan 80% data training dan 20% data testing dan melakukan scaling pada fitur.
6. Melakukan balancing data pada kolom target karena target terindikasi imbalanced sehingga harus diseimbangkan.

### Modeling

Beberapa model machine learning yang dilatih untuk memprediksi Dropout, Enrolled, & Graduate adalah sebagai berikut:

- **Logistik Regression**: untuk memahami kemungkinan mahasiswa dropout, enrolled atau graduate berdasarkan beberapa faktor.
- **SVM**: digunakan untuk memaksimalkan margin antara kelas dropout, enrolled, atau graduate dalam dataset yang memiliki dimensi tinggi.
- **Random Forest Classifier**: metode ensemble untuk prediksi yang lebih akurat dan mengurangi overfitting.

Berdasarkan model model yang dilatih, model Random Forest Classifier memberikan performa yang baik diantara model-model lainnya.

### Evaluation

Metrik evaluasi model yang digunakan dalam proyek ini meliputi:

- **Accuracy**: Untuk mengukur ketepatan prediksi model secara keseluruhan.
- **Precision & Recall**: Untuk memahami trade-off antara memprediksi positif yang benar dan positif yang salah.
- **F1-Score**: Keseimbangan antara presisi dan recall, terutama berguna dalam menangani ketidakseimbangan kelas.
- **Confusion Matrix**: Untuk memvisualisasikan kinerja dan mengidentifikasi area di mana model dapat salah mengklasifikasikan.

berikut adalah hasil evaluasi yang didapat pada model machine learning yang telah dilatih
| Model | Accuracy Training | Accuracy Testing |
|---------------|--------------|--------------|
| Logistic Regression | 73% | 74% |
| SVM | 77% | 73% |
| Random Forest | 100% | 75% |
| Random Forest (Hyperparameter Tuning) | 82% | 75% |

## Business Dashboard

Business dashboard yang telah dibuat menjadi satu dengan sistem machine learning. Anda dapat melihat dashboard yang telah dibuat dengan klik pada [Link ini](https://student-statuss.streamlit.app/) dan pilih pada side **dashboard**.

<center><img src="images\dashboard.png" alt="alt text" width="whatever" height="whatever"></center>
<center><img src="images\wahid_hasim-dashboard1.png" alt="alt text" width="whatever" height="whatever"></center>

## Menjalankan Sistem Machine Learning

1. Menyiapkan data: Pastikan dataset yang diperlukan berada di folder yang sesuai, atau tambahkan data baru yang ingin Anda prediksi.
2. Running Model: Untuk menjalankan model prediksi, gunakan script Python yang telah disediakan. Contoh untuk melakukan prediksi dengan model Neural Network:

```
streamlit run app.py
```

4. Atau bisa klik [Link ini](https://student-statuss.streamlit.app/) untuk mengakses aplikasi secara real-time dari streamlit dan pastikan memilih side **prediksi** seperti pada gambar dibawah.

<center><img src="images\prediksi.png" alt="alt text" width="whatever" height="whatever"></center>

## Conclusion:

Tujuan utama proyek ini adalah membangun model klasifikasi untuk memprediksi dropout, terdaftar dan keberhasilan akademik mahasiswa berdasarkan data demografi dan kinerja awal mereka. Berdasarkan hasil dari beberapa model, seperti Random Forest, Logistic Regression, Decision Tree, dan Neural Networks, kita telah mencapai pemodelan yang cukup baik dengan akurasi yang bervariasi. Hasil terbaik dicapai oleh model **Random Forest Classifier** dengan akurasi test **75%**, yang menunjukkan bahwa model dapat memprediksi dropout dan keberhasilan akademik mahasiswa dengan baik, namun dalam memprediksi terdaftar sedikit kurang bagus. Dengan faktor-faktor yang paling berpengaruh terhadap prediksi dropout, terdaftar dan keberhasilan akademik adalah `Approval_rate (15.37%)`, `Curricular_units_2nd_sem_approved (10.95%)`, &`Curricular_units_2nd_sem_grade (7.82%)`. Faktor-faktor ini menunjukkan pentingnya performa akademik di semester awal dalam menentukan keberhasilan, terdaftar ataupun dropout.

Secara keseluruhan, proyek ini sudah menjawab problem statement dan pertanyaan bisnis, serta berhasil mencapai sebagian besar tujuan yang diharapkan. Namun, ada ruang untuk meningkatkan performa model terutama dalam memprediksi dropout dengan lebih akurat. Model Random Forest dan Neural Networks adalah kandidat yang layak untuk diterapkan ke tahap deployment dengan beberapa penyempurnaan lebih lanjut.

### Rekomendation Action Items

Berdasarkan kesimpulan berikut adalah beberapa rekomendasi item yang dapat diterapkan oleh institusi pendidikan untuk menurunkan tingkat dropout dan meningkatkan keberhasilan akademik mahasiswa:

1. **Peningkatan Dukungan Akademik di Semester Awal**:
   Karena performa mahasiswa di semester awal (terutama pada mata kuliah yang diselesaikan) sangat mempengaruhi prediksi dropout, institusi dapat fokus memberikan dukungan tambahan di semester awal. Program seperti tutor tambahan, mentoring, dan bimbingan akademik khusus untuk mata kuliah yang sulit bisa menjadi strategi untuk membantu mahasiswa yang berisiko..

2. **Program Intervensi Berdasarkan Hasil Kinerja**:
   Institusi dapat mengimplementasikan program intervensi bagi mahasiswa yang mengalami kesulitan dalam menyelesaikan mata kuliah di semester awal atau yang memiliki admission grade rendah. Identifikasi dini berdasarkan data akademik dapat membantu mahasiswa sebelum mereka mengalami masalah yang lebih serius.
