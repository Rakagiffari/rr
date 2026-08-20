# =========================================
# utils/training.py
# =========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score

)


def train_model(X, y):

    # =====================================
    # PASTIKAN NUMERIK
    # =====================================

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    # =====================================
    # TRAIN TEST SPLIT
    # =====================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    # =====================================
    # RANDOM FOREST
    # =====================================

    rf = RandomForestClassifier(

        n_estimators=300,

        max_depth=15,

        min_samples_split=5,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    )

    # =====================================
    # TRAINING
    # =====================================

    rf.fit(
        X_train,
        y_train
    )

    # =====================================
    # PREDIKSI
    # =====================================

    y_pred = rf.predict(
        X_test
    )

    # =====================================
    # EVALUASI
    # =====================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(

        y_test,
        y_pred,

        zero_division=0

    )

    recall = recall_score(

        y_test,
        y_pred,

        zero_division=0

    )

    f1 = f1_score(

        y_test,
        y_pred,

        zero_division=0

    )

    report = classification_report(

        y_test,
        y_pred,

        zero_division=0

    )

    matrix = confusion_matrix(

        y_test,
        y_pred

    )

    # =====================================
    # FEATURE IMPORTANCE
    # =====================================

    importance = pd.DataFrame({

        "Fitur": X.columns,
        "Importance": rf.feature_importances_

    })

    grouped_importance = {

        "Indikasi": 0,
        "Km": 0,
        "Usia Motor": 0,
        "Jenis": 0
        
    }

    for _, row in importance.iterrows():

        fitur = row["Fitur"]
        nilai = row["Importance"]

        # NUMERIK

        if fitur == "Km":

            grouped_importance["Km"] += nilai

        elif fitur == "Usia Motor":

            grouped_importance["Usia Motor"] += nilai

        # KATEGORIKAL

        elif fitur.startswith("Jenis_"):

            grouped_importance["Jenis"] += nilai

        elif fitur.startswith("Indikasi_"):

            grouped_importance["Indikasi"] += nilai

    importance_grouped = pd.DataFrame({

        "Fitur": grouped_importance.keys(),
        "Importance": grouped_importance.values()

    })

    importance_grouped = importance_grouped.sort_values(

        by="Importance",
        ascending=False

    )

    importance_grouped = importance_grouped.reset_index(
        drop=True
    )

    # =====================================
    # RETURN
    # =====================================

    feature_names = X.columns.tolist()

    return (

        rf,

        accuracy,
        precision,
        recall,
        f1,

        report,
        matrix,

        importance_grouped,

        len(X_train),
        len(X_test),

        feature_names

    )

# ==========================================================
# KARAKTERISTIK HASIL KLASIFIKASI
# ==========================================================

def generate_vehicle_characteristics(model, X, feature_df):
    """
    Membuat ringkasan karakteristik kendaraan berdasarkan
    hasil prediksi Random Forest.
    """

    hasil = feature_df.copy()

    hasil["Prediksi"] = model.predict(X)

    # Jika hasil prediksi berupa angka
    if hasil["Prediksi"].dtype != object:

        hasil["Prediksi"] = hasil["Prediksi"].map({
            0: "Ringan",
            1: "Berat"
        })

    urutan_jenis = [
        "MAXi",
        "Classy",
        "Matic",
        "Moped",
        "Sport",
        "Off-road",
        "Unknown"
    ]

    summary = []

    for jenis in urutan_jenis:

        data_jenis = hasil[
            hasil["Jenis"] == jenis
        ]

        if data_jenis.empty:
            continue

        for service in ["Ringan", "Berat"]:

            data_service = data_jenis[
                data_jenis["Prediksi"] == service
            ]

            if data_service.empty:
                continue

            summary.append({

                "Jenis": jenis,

                "Service": service,

                "Indikasi Dominan":
                    data_service["Indikasi"]
                    .mode()
                    .iloc[0],

                "Rata-rata KM":
                    round(
                        data_service["Km"].mean(),
                        0
                    ),

                "Rata-rata Usia":
                    round(
                        data_service["Usia Motor"].mean(),
                        1
                    ),

                "Jumlah Data":
                    len(data_service)

            })

    return pd.DataFrame(summary)
