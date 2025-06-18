import streamlit as st
import pandas as pd
import joblib


# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        with open('model_graduation.pkl', 'rb') as file:
            model = joblib.load(file)
        return model
    except FileNotFoundError:
        st.error("File 'model_graduation.pkl' tidak ditemukan. Pastikan model berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Memuat model
nb_model = load_model()

# Judul aplikasi
st.title('Prediksi Kategori Waktu Kelulusan Mahasiswa')

st.write("""
Aplikasi ini memprediksi kategori waktu kelulusan (Tepat Waktu atau Terlambat) berdasarkan data akademik dan sosial mahasiswa.
""")

if nb_model is not None:
    # --- Input Data Baru ---
    st.header('Masukkan Data Baru untuk Prediksi')

    new_ACT = st.number_input('Masukkan nilai ACT composite score:', min_value=0.0, max_value=36.0, value=25.0)
    new_SAT = st.number_input('Masukkan nilai SAT total score:', min_value=0.0, max_value=1600.0, value=1200.0)
    new_GPA = st.number_input('Masukkan nilai rata-rata SMA (GPA):', min_value=0.0, max_value=4.0, value=3.5)
    new_income = st.number_input('Masukkan nilai pendapatan orang tua:', min_value=0.0, value=50000.0)
    new_education = st.number_input('Masukkan tingkat pendidikan orang tua (angka, misal: 1=SD, 2=SMP, 3=SMA, 4=S1, 5=S2+):', min_value=1.0, max_value=5.0, value=3.0, step=1.0)

    # Tombol untuk melakukan prediksi
    if st.button('Prediksi Kategori Kelulusan'):
        try:
            # Buat DataFrame dari input baru
            new_data_df = pd.DataFrame(
                [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
                columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
            )

            # Lakukan prediksi
            predicted_code = nb_model.predict(new_data_df)[0]

            # Konversi hasil prediksi ke label asli
            label_mapping = {1: 'Tepat Waktu (On Time)', 0: 'Terlambat (Late)'}
            predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

            st.success(f"**Prediksi kategori masa studi adalah: {predicted_label}**")

            st.write("---")
            st.subheader("Detail Input Anda:")
            st.write(new_data_df)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
else:
    st.warning("Model tidak dapat dimuat. Pastikan file 'model_graduation.pkl' sudah benar.")

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini dibuat untuk memprediksi kategori waktu kelulusan mahasiswa. "
    "Model yang digunakan adalah `model_graduation.pkl`."
)
