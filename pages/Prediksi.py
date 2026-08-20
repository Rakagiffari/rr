# ==========================================================
# IMPORT
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import io

from openpyxl import Workbook
from pathlib import Path
from datetime import datetime
from utils.preprocessing import preprocess_new_data
from utils.excel_handler import (
    simpan_layanan,
    baca_hari_ini,
    reset_hari_ini,
    update_status
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Analisis Data Baru",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# MODEL
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"

FEATURE_PATH = MODEL_DIR / "feature_names.pkl"

# ==========================================================
# STYLE
# ==========================================================

st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:700;
    color:white;
    margin-bottom:0px;
}

.sub-title{
    color:#9ca3af;
    font-size:14px;
    margin-top:-10px;
}

/* ===================================================
HASIL PREDIKSI
=================================================== */

.prediksi-card{
    border:1px solid #e5e7eb;
    border-radius:10px;
    padding:35px 20px;
    text-align:center;
    background:#f6fff8;
}

.prediksi-icon{
    font-size:58px;
    margin-bottom:10px;
}

.prediksi-title{
    font-size:18px;
    color:#555;
    margin-bottom:10px;
}

.prediksi-hasil{
    font-size:34px;
    font-weight:700;
    color:#0f9d58;
}

.prediksi-hasil-berat{
    color:#d32f2f;
}

.info-table{

    width:100%;
    border-collapse:collapse;

}

.info-table tr{

    border-bottom:1px solid #ececec;

}

.info-table td{

    padding:18px 0;

}

.info-left{

    color:#555;
    font-size:15px;

}

.info-right{

    text-align:right;
    font-size:18px;
    font-weight:700;

}

.badge-ringan{

    background:#d8f5dd;
    color:#0f9d58;
    padding:5px 14px;
    border-radius:20px;
    font-size:14px;
    font-weight:700;

}

.badge-berat{

    background:#ffdede;
    color:#d32f2f;
    padding:5px 14px;
    border-radius:20px;
    font-size:14px;
    font-weight:700;

}

..jadwal-box{
    border:1px solid #dcdcdc;
    border-radius:10px;
    background:#1f2937;
    padding:18px;
    text-align:center;
    height:150px;
}

.jadwal-judul{
    font-size:14px;
    color:#ffffff;
    margin-bottom:12px;
    font-weight:600;
}

.jadwal-isi{
    font-size:22px;
    font-weight:bold;
    color:#ffffff;
}

.jadwal-keterangan{
    margin-top:10px;
    color:#ffffff;
    font-size:13px;
}

.status-operasional{
    color:#16a34a;
}

.status-tutup{
    color:#dc2626;
}

/* ===================================================
SUCCESS BOX
=================================================== */

.stAlert{
    border-radius:8px;
}

.stAlert p{
    font-size:12.5px !important;
    font-weight:450 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <p class="main-title">
        📊 Analisis Data Baru
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="sub-title">
        Random Forest untuk Analisis dan Prediksi Layanan Service Kendaraan Yamaha
    </p>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# LANGKAH 1 - BAGIAN 2
# DATA KENDARAAN
# ==========================================================

with st.container(border=True):

    # ======================================================
    # MAPPING MODEL -> JENIS
    # ======================================================

    jenis_mapping = {

        # ================= MAXi =================
        "NMAX 155"         : "MAXi",
        "AEROX 155"        : "MAXi",
        "LEXI 125"         : "MAXi",
        "LEXI LX 155"      : "MAXi",
        "XMAX 250"         : "MAXi",
        "XMAX 300"         : "MAXi",
        "TMAX 560"         : "MAXi",

        # ================= Classy =================
        "FAZZIO 125"       : "Classy",
        "GRAND FILANO 125" : "Classy",

        # ================= Matic =================
        "MIO M3"           : "Matic",
        "MIO SPORTY"       : "Matic",
        "MIO J"            : "Matic",
        "MIO GT"           : "Matic",
        "MIO Z"            : "Matic",
        "MIO S"            : "Matic",
        "FREEGO 125"       : "Matic",
        "GEAR 125"         : "Matic",
        "FINO 125"         : "Matic",
        "FINO 115"         : "Matic",
        "XEON 125"         : "Matic",
        "SOUL GT 125"      : "Matic",
        "X-RIDE 125"       : "Matic",

        # ================= Moped =================
        "JUPITER Z"        : "Moped",
        "JUPITER MX"       : "Moped",
        "VEGA FORCE"       : "Moped",
        "CRYPTON"          : "Moped",

        # ================= Sport =================
        "VIXION"           : "Sport",
        "VIXION R"         : "Sport",
        "R15"              : "Sport",
        "R15M"             : "Sport",
        "R25"              : "Sport",
        "MT-15"            : "Sport",
        "XSR 155"          : "Sport",

        # ================= Off-road =================
        "WR155R"           : "Off-road",
        "YZ125X"           : "Off-road"

    }

    # ======================================================
    # BARIS 1
    # ======================================================

    col1, col2 = st.columns(2, gap="large")

    with col1:

        nama = st.text_input(
            "Nama Pelanggan",
            placeholder="Masukkan nama pelanggan"
        )

    with col2:

        no_polisi = st.text_input(
            "No. Polisi",
            placeholder="Contoh : BA 1234 XX"
        )

    # ======================================================
    # BARIS 2
    # ======================================================

    col3, col4 = st.columns(2, gap="large")

    with col3:

        model_motor = st.selectbox(

            "Model Motor",

            sorted(jenis_mapping.keys())

        )

    jenis_motor = jenis_mapping.get(
        model_motor,
        "-"
    )

    with col4:

        st.text_input(
            "Jenis",
            value=jenis_motor,
            disabled=True
        )

    # ======================================================
    # BARIS 3
    # ======================================================

    col5, col6 = st.columns(2, gap="large")

    with col5:

        kilometer = st.number_input(

            "Kilometer",

            min_value=0,

            value=0,

            step=500,

            format="%d"

        )

    with col6:

        tahun_motor = st.number_input(

            "Tahun Motor",

            min_value=1990,

            max_value=2100,

            value=2020,

            step=1,

            format="%d"

        )

    # ======================================================
    # BARIS 4
    # ======================================================

    indikasi = st.selectbox(

        "Indikasi Utama",

        [

            "Mesin",

            "Transmisi",

            "Sistem Bahan Bakar",

            "Kelistrikan",

            "Pengereman",

            "Roda dan Suspensi",

            "Body",

            "Umum"

        ]

    )

# ==========================================================
# TOMBOL PREDIKSI
# ==========================================================

left, center, right = st.columns([2, 2, 2])

with center:

    prediksi_button = st.button(
        "🔍 Prediksi Layanan",
        use_container_width=True,
        type="primary"
    )

# ==========================================================
# PROSES PREDIKSI RANDOM FOREST
# ==========================================================

if prediksi_button:

    try:

        # ==================================================
        # LOAD MODEL RANDOM FOREST
        # ==================================================

        model = joblib.load(MODEL_PATH)

        # ==================================================
        # PREPROCESS DATA INPUT
        # ==================================================

        X_new = preprocess_new_data(
            model_motor=model_motor,
            tahun_motor=tahun_motor,
            kilometer=kilometer,
            indikasi=indikasi
        )

        # ==================================================
        # PREDIKSI
        # ==================================================

        prediksi = model.predict(X_new)[0]

        probabilitas = model.predict_proba(X_new)[0]

        confidence = float(max(probabilitas) * 100)

        # ==================================================
        # KONVERSI LABEL
        # ==================================================

        # Sesuaikan jika label model berbeda
        if str(prediksi).lower() in ["ringan", "service ringan"]:
            hasil_prediksi = "SERVICE RINGAN"
            kategori = "Ringan"

        elif str(prediksi).lower() in ["berat", "service berat"]:
            hasil_prediksi = "SERVICE BERAT"
            kategori = "Berat"

        elif prediksi == 0:
            hasil_prediksi = "SERVICE RINGAN"
            kategori = "Ringan"

        else:
            hasil_prediksi = "SERVICE BERAT"
            kategori = "Berat"

        # ==================================================
        # SIMPAN KE SESSION STATE
        # ==================================================

        st.session_state["hasil_prediksi"] = hasil_prediksi
        st.session_state["kategori"] = kategori
        st.session_state["confidence"] = confidence

        st.session_state["nama"] = nama
        st.session_state["no_polisi"] = no_polisi
        st.session_state["model_motor"] = model_motor
        st.session_state["jenis_motor"] = jenis_motor
        st.session_state["kilometer"] = kilometer
        st.session_state["tahun_motor"] = tahun_motor
        st.session_state["indikasi"] = indikasi
    
        
        st.success("Prediksi berhasil dilakukan.")

    except Exception as e:

        st.error(f"Terjadi kesalahan saat melakukan prediksi.\n\n{e}")

# ==========================================================
# HASIL PREDIKSI
# ==========================================================

if "hasil_prediksi" in st.session_state:

    st.markdown("")

    with st.container(border=True):

        st.markdown("")

        col1, col2 = st.columns([1.15, 1.85], gap="large")

        # ==================================================
        # CARD HASIL PREDIKSI
        # ==================================================

        with col1:

            if st.session_state["kategori"] == "Ringan":

                st.markdown(
                    """
                    <h2 style="
                    text-align:center;
                    color:#16a34a;
                    margin-top:20px;
                    margin-bottom:20px;">
                    SERVICE RINGAN
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <h2 style="
                    text-align:center;
                    color:#dc2626;
                    margin-top:20px;
                    margin-bottom:20px;">
                    SERVICE BERAT
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

        # ==================================================
        # DETAIL HASIL
        # ==================================================

        with col2:

            df_hasil = pd.DataFrame({

                "Informasi": [

                    "🎯 Tingkat Keyakinan Model",
                    "🏷 Kategori Berdasarkan Model"

                ],

                "Hasil": [

                    f"{st.session_state['confidence']:.2f}%",
                    st.session_state["kategori"]

                ]

            })

            st.dataframe(

                df_hasil,

                hide_index=True,

                use_container_width=True

            )
