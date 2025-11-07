import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Konfigurasi Halaman & UI/UX Menarik ---
st.set_page_config(
    page_title="Prediksi Status Pinjaman (Decision Tree)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Utama & Tampilan Menarik
st.title("💸 Prediksi Kelayakan Pinjaman (Decision Tree)")
st.markdown("Aplikasi interaktif ini menggunakan **Model Decision Tree** yang dilatih dengan 5 fitur terpenting untuk memprediksi *loan status*.")
st.markdown("---")


# --- 1. Persiapan Data & Pelatihan Model ---
FILE_DATA = "loan_data.csv"
TARGET_COLUMN = 'loan_status' # 1: Default/Ditolak, 0: Dibayar/Diterima

# Top 5 Fitur Paling Penting (sesuai permintaan user)
FEATURES = ['previous_loan_defaults_on_file', 'loan_percent_income', 'loan_int_rate', 'person_income', 'person_home_ownership']

try:
    df = pd.read_csv(FILE_DATA)
    df_clean = df.copy()

    # --- Preprocessing ---
    # 1. Mengubah target kategorikal menjadi numerik jika belum (hanya untuk memastikan)
    if df_clean[TARGET_COLUMN].dtype == 'object':
        le = LabelEncoder()
        df_clean[TARGET_COLUMN] = le.fit_transform(df_clean[TARGET_COLUMN])

    # 2. Definisi jenis kolom
    numerical_features = ['loan_percent_income', 'loan_int_rate', 'person_income']
    binary_features = ['previous_loan_defaults_on_file'] # 'Yes'/'No'
    categorical_features = ['person_home_ownership'] # RENT, OWN, MORTGAGE, OTHER

    # 3. Handling NaN (Imputasi)
    # loan_int_rate sering memiliki NaN. Impute dengan mean.
    imputer = SimpleImputer(strategy='mean')
    df_clean['loan_int_rate'] = imputer.fit_transform(df_clean[['loan_int_rate']])

    # 4. Membuat Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_features), # Imputer for number
            ('bin', LabelEncoder(), binary_features), # Encoder for Yes/No
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # OneHot for home ownership
        ],
        remainder='drop' # Hanya gunakan fitur yang didefinisikan
    )

    # Khusus untuk binary 'previous_loan_defaults_on_file', kita harus mengkodekan secara manual di luar CT
    # atau menggunakan LabelEncoder di sini:
    df_clean['previous_loan_defaults_on_file'] = df_clean['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    
    # Memilih X dan y
    X = df_clean[FEATURES]
    y = df_clean[TARGET_COLUMN]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Re-Define Preprocessor for Simplicity (Manual Fit & Transform) ---
    # Karena Decision Tree bisa menangani biner/numerik, kita lakukan encoding dan imputasi di luar pipeline untuk kemudahan Streamlit
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Imputasi Bunga (Int Rate)
    mean_int_rate = X_train_processed['loan_int_rate'].mean()
    X_train_processed['loan_int_rate'].fillna(mean_int_rate, inplace=True)
    X_test_processed['loan_int_rate'].fillna(mean_int_rate, inplace=True)

    # Encoding 'previous_loan_defaults_on_file'
    X_train_processed['previous_loan_defaults_on_file'] = X_train_processed['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    X_test_processed['previous_loan_defaults_on_file'] = X_test_processed['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding 'person_home_ownership'
    X_train_processed = pd.get_dummies(X_train_processed, columns=['person_home_ownership'], drop_first=True)
    X_test_processed = pd.get_dummies(X_test_processed, columns=['person_home_ownership'], drop_first=True)
    
    # Menyinkronkan kolom (penting setelah OHE)
    common_cols = list(set(X_train_processed.columns) & set(X_test_processed.columns))
    X_train_final = X_train_processed[common_cols]
    X_test_final = X_test_processed[common_cols]
    
    # Inisialisasi dan Latih Model Decision Tree
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train_final, y_train)
    
    # Evaluasi Model
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1]

    # Metrik Klasifikasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Menyimpan daftar kolom akhir untuk input prediksi di Tab 1
    model_features = X_train_final.columns.tolist()
    global_mean_int_rate = df_clean['loan_int_rate'].mean()

    st.sidebar.success(f"Model Decision Tree berhasil dilatih dengan {len(model_features)} fitur.")

except FileNotFoundError:
    st.error(f"⚠️ Error: File '{FILE_DATA}' tidak ditemukan. Pastikan file berada dalam direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan saat memproses data atau melatih model: {e}")
    st.stop()


# --- 2. Implementasi Tab UI/UX ---
tab1, tab2 = st.tabs(["💡 Prediksi Loan Status", "📊 Analisis Data & Grafik"])

with tab1:
    st.header("Form Input Prediksi Pinjaman")
    st.markdown("Masukkan data 5 fitur penting pinjaman untuk mendapatkan prediksi status dari **Model Decision Tree**.")
    
    st.subheader("Top 5 Fitur Utama (Decision Tree)")
    
    # Mengatur layout input form dalam kolom
    col_input_1, col_input_2 = st.columns(2)

    with col_input_1:
        # person_income (pendapatan)
        person_income = st.number_input("Pendapatan Tahunan (person_income)", min_value=10000, max_value=300000, value=75000, step=1000)
        # loan_int_rate (bunga pinjaman)
        loan_int_rate = st.number_input("Bunga Pinjaman (%) (loan_int_rate)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
        # previous_loan_defaults_on_file
        previous_loan_defaults_on_file = st.selectbox("Pernah Gagal Bayar Sebelumnya?", options=['No', 'Yes'])


    with col_input_2:
        # loan_percent_income (% pinjaman dari pendapatan)
        loan_percent_income = st.slider("% Pinjaman dari Pendapatan (loan_percent_income)", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
        # person_home_ownership
        person_home_ownership = st.selectbox("Status Kepemilikan Rumah", options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        
        st.info("Input ini adalah 5 fitur yang paling memengaruhi hasil prediksi Model Decision Tree.")

    # Tombol Prediksi
    if st.button("Prediksi Status Pinjaman (Decision Tree)", help="Klik untuk mendapatkan hasil prediksi model."):
        
        # --- Preprocessing Data Input Baru ---
        input_data = {
            'loan_percent_income': loan_percent_income,
            'loan_int_rate': loan_int_rate,
            'person_income': person_income,
            # Fitur yang akan di-OHE dan Biner
            'person_home_ownership': person_home_ownership,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }
        
        new_data_df = pd.DataFrame([input_data])
        
        # 1. Encoding Biner
        new_data_df['previous_loan_defaults_on_file'] = new_data_df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
        
        # 2. One-Hot Encoding Kepemilikan Rumah
        input_processed = new_data_df.copy()
        
        # Tambahkan semua kolom OHE yang mungkin ada dari pelatihan, setel ke 0
        for col in [c for c in model_features if c.startswith('person_home_ownership_')]:
             input_processed[col] = 0

        # Setel kolom OHE yang sesuai dari input pengguna menjadi 1
        col_name = f'person_home_ownership_{person_home_ownership}'
        if col_name in input_processed.columns:
            input_processed[col_name] = 1

        # Drop kolom asli dan ambil hanya kolom yang digunakan model
        input_final = input_processed.drop(columns=['person_home_ownership'])[model_features]

        # Melakukan Prediksi
        prediction = model.predict(input_final)[0]
        prediction_proba = model.predict_proba(input_final)[0][1] # Probabilitas default (kelas 1)

        st.subheader("✅ Hasil Prediksi")
        
        if prediction == 0:
            st.success(f"**Status Pinjaman Diprediksi: DITERIMA/DIBAYAR**")
            st.metric(label="Probabilitas Default (Kelas 1)", value=f"{prediction_proba*100:.2f}%")
        else:
            st.error(f"**Status Pinjaman Diprediksi: DEFAULT/DITOLAK**")
            st.metric(label="Probabilitas Default (Kelas 1)", value=f"{prediction_proba*100:.2f}%")
            
        st.markdown(f"*(**Model yang digunakan: Decision Tree Classifier**)*")


with tab2:
    st.header("Analisis Data & Evaluasi Model Decision Tree")
    st.markdown("Tab ini menampilkan kinerja Model **Decision Tree** menggunakan metrik klasifikasi standar.")
    
    st.subheader("Metrik Kinerja Model Decision Tree")
    
    # Tampilkan Metrik
    col_metrics = st.columns(5)
    
    col_metrics[0].metric(label="Akurasi", value=f"{accuracy:.4f}", help="Proporsi prediksi benar secara keseluruhan.")
    col_metrics[1].metric(label="Precision", value=f"{precision:.4f}", help="Ketepatan model dalam memprediksi kelas positif (1).")
    col_metrics[2].metric(label="Recall", value=f"{recall:.4f}", help="Kemampuan model menemukan semua kasus positif (1).")
    col_metrics[3].metric(label="F1-Score", value=f"{f1:.4f}", help="Rata-rata harmonik dari Precision dan Recall.")
    col_metrics[4].metric(label="ROC-AUC Score", value=f"{roc_auc:.4f}", help="Area di bawah kurva ROC. Mengukur kemampuan diskriminasi model.")
    
    st.markdown(f"*(Semua metrik di atas dihitung berdasarkan hasil Model **Decision Tree**)*")
    st.markdown("---")
    
    st.subheader("Visualisasi Data Utama")

    # Visualisasi 1: Matriks Kebingungan (Confusion Matrix)
    st.markdown("#### 1. Matriks Kebingungan (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dibayar (0)', 'Default (1)'], 
                yticklabels=['Dibayar (0)', 'Default (1)'], ax=ax1)
    ax1.set_title('Confusion Matrix (Decision Tree)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    st.pyplot(fig1)
    plt.close(fig1) # Pembersihan plot

    # Visualisasi 2: Distribusi Loan % Income
    st.markdown("#### 2. Distribusi Loan % Income berdasarkan Status Pinjaman")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_clean, x='loan_percent_income', hue=TARGET_COLUMN, kde=True, ax=ax2, palette='viridis')
    ax2.set_title('Distribusi loan_percent_income (Decision Tree)')
    ax2.legend(title='Loan Status', labels=['Default/Ditolak (1)', 'Dibayar/Diterima (0)'])
    st.pyplot(fig2)
    plt.close(fig2) # Pembersihan plot
    
    # Visualisasi 3: Count Plot Kepemilikan Rumah vs Loan Status
    st.markdown("#### 3. Count Plot Kepemilikan Rumah vs Status Pinjaman")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='person_home_ownership', hue=TARGET_COLUMN, data=df_clean, ax=ax3, palette='Set2')
    ax3.set_title('Person Home Ownership vs Loan Status (Decision Tree)')
    ax3.set_xlabel('Kepemilikan Rumah')
    ax3.set_ylabel('Jumlah Data')
    ax3.legend(title='Loan Status', labels=['Dibayar/Diterima (0)', 'Default/Ditolak (1)'])
    st.pyplot(fig3)
    plt.close(fig3) # Pembersihan plot
