# =========================================
# utils/preprocessing.py
# =========================================

import pandas as pd
from datetime import datetime


# =========================================
# PEMBENTUKAN JENIS MOTOR
# =========================================

def get_jenis(model):

    model = str(model).upper()

    # MAXI
    if any(x in model for x in [
        "XMAX",
        "NMAX",
        "AEROX",
        "LEXI",
        "TMAX"
    ]):
        return "MAXi"

    # CLASSY
    elif any(x in model for x in [
        "FAZZIO",
        "FILANO"
    ]):
        return "Classy"

    # MATIC
    elif any(x in model for x in [
        "MIO",
        "SOUL",
        "XEON",
        "FINO",
        "GEAR",
        "FREEGO",
        "X-RIDE",
        "XRIDE",
        "NOUVO",
        "LEXAM"
    ]):
        return "Matic"

    # SPORT
    elif any(x in model for x in [
        "R15",
        "R25",
        "R6",
        "R1",
        "VIXION",
        "BYSON",
        "SCORPIO",
        "RX",
        "XSR",
        "MT"
    ]):
        return "Sport"

    # OFFROAD
    elif any(x in model for x in [
        "WR",
        "YZ"
    ]):
        return "Off-road"

    # MOPED
    elif any(x in model for x in [
        "JUPITER",
        "VEGA",
        "CRYPTON",
        "ALFA",
        "SIGMA",
        "F1ZR",
        "MX KING"
    ]):
        return "Moped"

    return "Unknown"


# =========================================
# PREPROCESSING
# =========================================

def preprocess_data(df):

    # =====================================
    # HAPUS DUPLIKAT
    # =====================================

    df = df.drop_duplicates()

    # =====================================
    # VALIDASI KOLOM
    # =====================================

    required_columns = [

        "Model",
        "Tahun",
        "Km",
        "Indikasi",
        "Service"

    ]

    missing_columns = [

        col for col in required_columns
        if col not in df.columns

    ]

    if missing_columns:

        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    # =====================================
    # HANDLE MISSING VALUE
    # =====================================
    
    df["Model"] = df["Model"].fillna(
        "Unknown"
    )

    df["Indikasi"] = df["Indikasi"].fillna(
        "Unknown"
    )

    # =====================================
    # NUMERIK
    # =====================================

    df["Tahun"] = pd.to_numeric(
        df["Tahun"],
        errors="coerce"
    )

    df["Km"] = pd.to_numeric(
        df["Km"],
        errors="coerce"
    )

    df["Tahun"] = df["Tahun"].fillna(
        df["Tahun"].median()
    )

    df["Km"] = df["Km"].fillna(
        df["Km"].median()
    )

    # =====================================
    # FEATURE ENGINEERING
    # =====================================

    tahun_sekarang = datetime.now().year

    df["Usia Motor"] = (
        tahun_sekarang - df["Tahun"]
    )

    df["Jenis"] = df["Model"].apply(
        get_jenis
    )

    # =====================================
    # TARGET
    # =====================================

    y = df["Service"].map({

        "Ringan": 0,
        "Berat": 1

    })

    if y.isnull().sum() > 0:

        raise ValueError(
            "Kolom Service harus berisi Ringan atau Berat"
        )

    # =====================================
    # FITUR FINAL
    # =====================================

    X = df[

        [
            "Jenis",
            "Km",
            "Usia Motor",
            "Indikasi"
        ]

    ].copy()

    # =====================================
    # ONE HOT ENCODING
    # =====================================

    X = pd.get_dummies(

        X,

        columns=[
            "Jenis",
            "Indikasi"
        ],

        drop_first=False

    )

    # =====================================
    # PASTIKAN NUMERIK
    # =====================================

    X = X.astype(float)

    return X, y

# =========================================
# PREPROCESS DATA BARU (UNTUK PREDIKSI)
# =========================================

import joblib
from pathlib import Path


def preprocess_new_data(
    model_motor,
    tahun_motor,
    kilometer,
    indikasi
):
    """
    Mengubah input pengguna menjadi format yang sama
    seperti data training Random Forest.
    """

    # ==============================
    # HITUNG USIA MOTOR
    # ==============================

    tahun_sekarang = datetime.now().year

    usia_motor = tahun_sekarang - tahun_motor

    # ==============================
    # TENTUKAN JENIS MOTOR
    # ==============================

    jenis = get_jenis(model_motor)

    # ==============================
    # MEMBUAT DATAFRAME
    # ==============================

    X_new = pd.DataFrame({

        "Jenis": [jenis],

        "Km": [kilometer],

        "Usia Motor": [usia_motor],

        "Indikasi": [indikasi]

    })

    # ==============================
    # ONE HOT ENCODING
    # ==============================

    X_new = pd.get_dummies(

        X_new,

        columns=[
            "Jenis",
            "Indikasi"
        ],

        drop_first=False

    )

    # ==============================
    # LOAD FEATURE NAMES
    # ==============================

    BASE_DIR = Path(__file__).resolve().parent.parent

    feature_names = joblib.load(
        BASE_DIR / "model" / "feature_names.pkl"
    )

    # ==============================
    # TAMBAHKAN KOLOM YANG BELUM ADA
    # ==============================

    for col in feature_names:

        if col not in X_new.columns:

            X_new[col] = 0

    # ==============================
    # HAPUS KOLOM YANG TIDAK DIGUNAKAN
    # ==============================

    X_new = X_new[feature_names]

    # ==============================
    # PASTIKAN NUMERIK
    # ==============================

    X_new = X_new.astype(float)

    return X_new
