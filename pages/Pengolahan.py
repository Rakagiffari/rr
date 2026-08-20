import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import matplotlib.patches as patches
import graphviz

from pathlib import Path

from utils.preprocessing import preprocess_data
from utils.training import (
    train_model,
    generate_vehicle_characteristics
)
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    _tree
)
from sklearn.tree import export_graphviz
from utils.report import generate_pdf

# =========================================
# CONFIG
# =========================================

st.set_page_config(
    page_title="Training Model",
    page_icon="⚙️",
    layout="wide"
)

# =========================================
# STYLE
# =========================================

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

.metric-card{
    background:#1f2937;
    padding:20px;
    border-radius:16px;
    text-align:center;
}

.metric-label{
    color:#9ca3af;
    font-size:15px;
}

.metric-value{
    color:white;
    font-size:30px;
    font-weight:bold;
}

/* =========================================
UPLOAD INFORMATION CARD
========================================= */

.upload-card{
    background:linear-gradient(145deg,#111827,#1e293b);
    border:1px solid #334155;
    border-radius:18px;
    padding:22px;
    text-align:center;
    transition:.3s;
    box-shadow:0 5px 15px rgba(0,0,0,.08);
    min-height:120px;

    display:flex;
    flex-direction:column;
    justify-content:center;
}

.upload-card:hover{
    transform:translateY(-5px);
    box-shadow:0 0 18px rgba(239,68,68,.25);
}

.upload-icon{
    font-size:30px;
    margin-bottom:8px;
}

.upload-title{
    font-size:14px;
    color:#94a3b8;
    font-weight:600;
    margin-bottom:8px;
}

.upload-value{
    font-size:22px;
    color:white;
    font-weight:800;
    word-break:break-word;
}

/* =========================================
EXPANDER
========================================= */

div[data-testid="stExpander"]{
    border:1px solid #334155;
    border-radius:12px;
    background:#020617;
    box-shadow:0 2px 8px rgba(0,0,0,.05);
    margin-bottom:5px;
}

div[data-testid="stExpander"] details summary{
    font-size:16px;
    font-weight:600;
    padding:12px 18px;
}

div[data-testid="stExpander"] details summary:hover{
    background:#1f2937;
    border-radius:12px;
}

div[data-testid="stExpanderDetails"]{
    padding:15px 18px 18px 18px;
}

/* ===================================================
   SUCCESS BOX
=================================================== */
.stAlert{
    border-radius: 8px;
}
.stAlert p{
    font-size: 12.5px !important;
    font-weight: 450 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# FUNGSI REPRESENTATIVE DECISION TREE
# ==========================================================

def extract_tree_paths(tree_model, feature_names):
    tree = tree_model.tree_
    paths = []
    
    def recurse(node, current_path):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            recurse(
                tree.children_left[node],
                current_path + [
                    f"{feature} ≤ {threshold:.2f}"
                ]
            )

            recurse(
                tree.children_right[node],
                current_path + [
                    f"{feature} > {threshold:.2f}"
                ]
            )

        else:
            samples = tree.n_node_samples[node]
            values = tree.value[node][0]
            label = values.argmax()
            total = values.sum()
            
            purity = (
                values[label] / total * 100
                if total > 0
                else 0
            )

            prediction = (
                "Service Ringan"
                if label == 0
                else "Service Berat"
            )

            paths.append({
                "path": current_path,
                "prediction": prediction,
                "samples": samples,
                "purity": purity
            })

    recurse(0, [])

    paths = sorted(
        paths,
        key=lambda x: (
            x["samples"],
            x["purity"]
        ),
        reverse=True
    )

    return paths

# ==========================================================
# FUNGSI KARAKTERISTIK KENDARAAN
# ==========================================================

def build_characteristics(path_rules):

    characteristics = []

    for rule in path_rules:

        if ">" not in rule:
            continue

        feature = rule.split(">")[0].strip()
        value = rule.split(">")[1].strip()

        # ---------------- Jenis ----------------
        if feature.startswith("Jenis_"):

            jenis = feature.replace("Jenis_", "")

            characteristics.append(
                f"Jenis motor **{jenis}**"
            )

        # ---------------- Indikasi ----------------
        elif feature.startswith("Indikasi_"):

            indikasi = feature.replace("Indikasi_", "")

            characteristics.append(
                f"Indikasi **{indikasi}**"
            )

        # ---------------- Kilometer ----------------
        elif feature == "Km":

            characteristics.append(
                f"Kilometer lebih dari **{float(value):,.0f} km**".replace(",", ".")
            )

        # ---------------- Usia ----------------
        elif feature == "Usia Motor":

            characteristics.append(
                f"Usia motor lebih dari **{float(value):.0f} tahun**"
            )

    return list(dict.fromkeys(characteristics))
    
# =========================================
# HEADER
# =========================================

st.markdown(
    '<p class="main-title">⚙️ Pengolahan Data</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Random Forest untuk Klasifikasi Layanan Service Kendaraan Yamaha</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "📂 Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

# =========================================
# MAIN
# =========================================

if uploaded_file is not None:

    try:

        # =====================================
        # MEMBACA DATASET
        # =====================================

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith((".xlsx", ".xls")):

            df = pd.read_excel(uploaded_file)

        else:

            st.error("Format file tidak didukung.")
            st.stop()

        st.success("Upload Data berhasil")

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

            col
            for col in required_columns
            if col not in df.columns

        ]

        if missing_columns:

            st.error(
                f"Kolom berikut tidak ditemukan:\n{', '.join(missing_columns)}"
            )

            st.stop()

        # =====================================
        # INFORMASI DATASET
        # =====================================
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="upload-card">
                <div class="upload-title">
                    Nama File
                </div>
                <div class="upload-value">
                    {uploaded_file.name}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="upload-card">
                <div class="upload-title">
                    Jumlah Data
                </div>
                <div class="upload-value">
                    {len(df):,}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="upload-card">
                <div class="upload-title">
                    Jumlah Kolom
                </div>
                <div class="upload-value">
                    {len(df.columns)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            
        # =====================================
        # PREVIEW DATASET
        # =====================================

        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("Preview Dataset", expanded=False):
            st.caption(f"Menampilkan seluruh dataset ({len(df)} baris).")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=500
            )

        # =====================================
        # TIPE DATA
        # =====================================
        with st.expander("Tipe Data", expanded=False):
            st.caption("Ringkasan tipe data pada dataset.")
            numeric_cols = len(
                df.select_dtypes(include=["number"]).columns
            )

            categorical_cols = len(
                df.select_dtypes(include=["object", "category"]).columns
            )

            tipe_df = pd.DataFrame({
                "Nama": [
                    "Numerik",
                    "Kategori"
                ],

                "Jumlah": [
                    numeric_cols,
                    categorical_cols
                ]
            })

            st.dataframe(
                tipe_df,
                use_container_width=True,
                hide_index=True
            )

        # =====================================
        # MISSING VALUE
        # =====================================

        with st.expander("Missing Value", expanded=False):

            st.caption("Pemeriksaan Missing Value pada seluruh dataset.")

            total_missing = df.isnull().sum().sum()

            missing_df = (
                df.isnull()
                .sum()
                .reset_index()
            )

            missing_df.columns = [
                "Kolom",
                "Jumlah Missing"
            ]

            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )

        # =====================================
        # DATA DUPLIKAT
        # =====================================

        with st.expander("Data Duplikat", expanded=False):

            st.caption(
                "Ringkasan pemeriksaan data duplikat pada dataset."
            )

            total_duplicate = df.duplicated().sum()

            duplicate_info = pd.DataFrame({

                "Nama": [
                    "Jumlah Data",
                    "Data Duplikat"
                ],

                "Jumlah": [
                    len(df),
                    total_duplicate
                ]

            })

            st.dataframe(
                duplicate_info,
                use_container_width=True,
                hide_index=True
            )

        # =====================================
        # DISTRIBUSI DATA
        # =====================================

        with st.expander("Distribusi Data", expanded=False):

            st.caption(
                "Distribusi jumlah data pada setiap kategori layanan service."
            )

            service_count = df["Service"].value_counts()

            # =====================================
            # VISUALISASI
            # =====================================

            left, col_bar, space, col_pie, right = st.columns(
                [0.5, 3, 0.3, 3, 0.5]
            )

            # -------------------------------------
            # BAR CHART
            # -------------------------------------

            fig_bar, ax_bar = plt.subplots(
                figsize=(3.8, 3.2),
                dpi=120
            )

            sns.barplot(
                x=service_count.index,
                y=service_count.values,
                palette="Reds",
                ax=ax_bar
            )

            ax_bar.set_title(
                "Distribusi Target",
                fontsize=10
            )

            ax_bar.set_xlabel(
                "Kategori",
                fontsize=8
            )

            ax_bar.set_ylabel(
                "Jumlah",
                fontsize=8
            )

            ax_bar.tick_params(
                axis="both",
                labelsize=8
            )

            ymax = service_count.max()

            for i, value in enumerate(service_count.values):

                ax_bar.text(
                    i,
                    value + (ymax * 0.02),
                    str(value),
                    ha="center",
                    fontsize=8,
                    fontweight="bold"
                )

            plt.tight_layout()

            # -------------------------------------
            # PIE CHART
            # -------------------------------------

            fig_pie, ax_pie = plt.subplots(
                figsize=(3.8, 3.2),
                dpi=120
            )

            colors = sns.color_palette(
                "Reds",
                len(service_count)
            )

            ax_pie.pie(
                service_count.values,
                labels=service_count.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                textprops={
                    "fontsize": 8
                },
                wedgeprops={
                    "edgecolor": "white"
                }
            )

            ax_pie.set_title(
                "Persentase Target",
                fontsize=10
            )

            ax_pie.axis("equal")

            plt.tight_layout()

            # -------------------------------------
            # TAMPILKAN BERSEBELAHAN
            # -------------------------------------

            with col_bar:

                st.pyplot(
                    fig_bar,
                    use_container_width=False
                )

            with col_pie:

                st.pyplot(
                    fig_pie,
                    use_container_width=False
                )

            plt.close(fig_bar)
            plt.close(fig_pie)

            # =====================================
            # SIMPAN VISUALISASI UNTUK PDF
            # =====================================

            BASE_DIR = Path(__file__).parent.parent

            distribution_path = (
                BASE_DIR /
                "distribution_target.png"
            )

            fig_pdf, (ax_pdf_bar, ax_pdf_pie) = plt.subplots(
                1,
                2,
                figsize=(8, 3.8),
                dpi=220
            )

            # -------------------------------------
            # BAR CHART
            # -------------------------------------

            sns.barplot(
                x=service_count.index,
                y=service_count.values,
                palette="Reds",
                ax=ax_pdf_bar
            )

            ax_pdf_bar.set_title(
                "Distribusi Target",
                fontsize=10
            )

            ax_pdf_bar.set_xlabel(
                "Kategori",
                fontsize=8
            )    

            ax_pdf_bar.set_ylabel(
                "Jumlah",
                fontsize=8
            )

            ax_pdf_bar.tick_params(
                axis="both",
                labelsize=8
            )

            ymax = service_count.max()

            for i, value in enumerate(service_count.values):

                ax_pdf_bar.text(
                    i,
                    value + (ymax * 0.02),
                    str(value),
                    ha="center",
                    fontsize=8,
                    fontweight="bold"
                )

            # -------------------------------------
            # PIE CHART
            # -------------------------------------

            colors = sns.color_palette(
                "Reds",
                len(service_count)
            )

            ax_pdf_pie.pie(
                service_count.values,
                labels=service_count.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                textprops={
                    "fontsize": 8
                },
                wedgeprops={
                    "edgecolor": "white"
                }
            )

            ax_pdf_pie.set_title(
                "Persentase Target",
                fontsize=10
            )

            ax_pdf_pie.axis("equal")

            plt.tight_layout()

            fig_pdf.savefig(
                distribution_path,
                dpi=250,
                bbox_inches="tight"
            )

            plt.close(fig_pdf)

            st.markdown("---")

            # =====================================
            # TABEL DISTRIBUSI
            # =====================================

            distribusi_df = pd.DataFrame({

                "Kategori Service": service_count.index,

                "Jumlah Data": service_count.values,

                "Persentase (%)": (
                    service_count.values /
                    len(df) *
                    100
                ).round(2)

            })

            st.dataframe(
                distribusi_df,
                use_container_width=True,
                hide_index=True
            )
        # =====================================
        # FEATURE ENGINEERING
        # =====================================

        # Salinan dataset hanya untuk visualisasi
        feature_df = df.copy()

        from datetime import datetime

        tahun_sekarang = datetime.now().year
        
        # -------------------------------------
        # Feature 1 : Usia Motor
        # -------------------------------------

        feature_df["Usia Motor"] = (
            tahun_sekarang - feature_df["Tahun"]
        )

        # -------------------------------------
        # Feature 2 : Jenis Motor
        # -------------------------------------

        def get_jenis(model):

            model = str(model).upper()

            if any(x in model for x in [
                "XMAX", "NMAX", "AEROX", "LEXI", "TMAX"
            ]):
                return "MAXi"

            elif any(x in model for x in [
                "FAZZIO", "FILANO"
            ]):
                return "Classy"

            elif any(x in model for x in [
                "MIO", "SOUL", "XEON", "FINO",
                "GEAR", "FREEGO", "X-RIDE",
                "XRIDE", "NOUVO", "LEXAM"
            ]):
                return "Matic"

            elif any(x in model for x in [
                "R15", "R25", "R6", "R1",
                "VIXION", "BYSON",
                "SCORPIO", "RX",
                "XSR", "MT"
            ]):
                return "Sport"

            elif any(x in model for x in [
                "WR", "YZ"
            ]):
                return "Off-road"

            elif any(x in model for x in [
                "JUPITER", "VEGA",
                "CRYPTON", "ALFA",
                "SIGMA", "F1ZR",
                "MX"
            ]):
                return "Moped"

            return "Unknown"

        feature_df["Jenis"] = feature_df["Model"].apply(get_jenis)

        # -------------------------------------
        # Atur posisi kolom
        # -------------------------------------

        cols = feature_df.columns.tolist()

        cols.remove("Jenis")
        cols.remove("Usia Motor")

        cols.insert(cols.index("Model") + 1, "Jenis")
        cols.insert(cols.index("Tahun") + 1, "Usia Motor")

        feature_df = feature_df[cols]

        # -------------------------------------
        # TAMPILAN
        # -------------------------------------

        with st.expander("Feature Engineering", expanded=False):

            st.caption(
                "Proses penambahan fitur."
            )

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True,
                height=350
            )

        # =====================================
        # PREPROCESS DATA
        # =====================================

        X, y = preprocess_data(df)
        
        # =====================================
        # TRAINING MODEL
        # =====================================

        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:

            train_button = st.button(
                "🚀 Mulai Training Model",
                use_container_width=True
            )

        st.markdown("""
            <div style="
                text-align:center;
                color:#9ca3af;
                font-size:14px;
                margin-top:-8px;
                margin-bottom:20px;
            ">
                Tekan tombol berikut untuk melatih model Random Forest menggunakan dataset yang telah diproses.
            </div>
        """, unsafe_allow_html=True)

        if train_button:

            import time

            progress = st.empty()

            # =====================================
            # ENCODING
            # =====================================

            progress.info("Melakukan encoding...")
            time.sleep(0.8)

            progress.success("Encoding selesai")
            time.sleep(0.3)

            # =====================================
            # TRAIN TEST SPLIT
            # =====================================

            progress.info("Membagi data train dan test...")
            time.sleep(0.8)

            progress.success("Train-test split selesai")
            time.sleep(0.3)

            # =====================================
            # TRAIN RANDOM FOREST
            # =====================================

            progress.info("Melatih Random Forest...")

            (
                model,
                accuracy,
                precision,
                recall,
                f1,
                report,
                matrix,
                importance_grouped,
                train_size,
                test_size,
                feature_names
            ) = train_model(X, y)

            # =====================================
            # KARAKTERISTIK HASIL KLASIFIKASI
            # =====================================

            summary_df = generate_vehicle_characteristics(
                model,
                X,
                feature_df
            )

            progress.success("Model berhasil dilatih")
            time.sleep(0.5)

            # =====================================
            # EVALUASI
            # =====================================

            progress.info("Menghitung evaluasi...")
            time.sleep(0.8)

            progress.success("Evaluasi selesai")
            time.sleep(0.3)

            # =====================================
            # INSIGHT
            # =====================================

            progress.info("Menyusun insight...")
            time.sleep(0.8)

            progress.success("Insight berhasil dibuat")
            time.sleep(0.3)

            # Hilangkan animasi
            progress.empty()

            # =====================================
            # SIMPAN MODEL
            # =====================================

            BASE_DIR = Path(__file__).parent.parent

            model_dir = BASE_DIR / "model"

            model_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            joblib.dump(
                model,
                model_dir / "random_forest_model.pkl"
            )

            joblib.dump(
                feature_names,
                model_dir / "feature_names.pkl"
            )

            # =====================================
            # CLASSIFICATION REPORT
            # =====================================

            with st.expander("Classification Report", expanded=False):

                st.caption(
                    "Hasil evaluasi model Random Forest berdasarkan Precision, Recall, F1-Score, dan Support."
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.markdown(f"""
                    <div class="upload-card">
                        <div class="upload-title">
                            Accuracy
                        </div>
                        <div class="upload-value">
                            {accuracy:.2%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:

                    st.markdown(f"""
                    <div class="upload-card">
                        <div class="upload-title">
                            Precision
                        </div>
                        <div class="upload-value">
                            {precision:.2%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c3:

                    st.markdown(f"""
                    <div class="upload-card">
                        <div class="upload-title">
                            Recall
                        </div>
                        <div class="upload-value">
                            {recall:.2%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c4:

                    st.markdown(f"""
                    <div class="upload-card">
                        <div class="upload-title">
                            F1-Score
                        </div>
                        <div class="upload-value">
                            {f1:.2%}
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

                st.code(report)
            
            # =====================================
            # CONFUSION MATRIX
            # =====================================

            # Membuat visualisasi confusion matrix
            fig2, ax2 = plt.subplots(
                figsize=(3.8, 3.2),
                dpi=120
            )

            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Reds",
                cbar=False,
                xticklabels=["Ringan", "Berat"],
                yticklabels=["Ringan", "Berat"],
                annot_kws={
                    "size": 10,
                },
                linewidths=0.5,
                linecolor="white",
                square=True,
                ax=ax2
            )

            ax2.set_xlabel(
                "Prediksi",
                fontsize=8
            )

            ax2.set_ylabel(
                "Aktual",
                fontsize=8
            )

            ax2.tick_params(
                axis="both",
                labelsize=8
            )

            plt.tight_layout()

            # Tampilkan di tengah halaman
            kiri, tengah, kanan = st.columns([2, 3, 2])

            with tengah:
                st.pyplot(
                    fig2,
                    use_container_width=False
                )

            # Simpan untuk PDF
            cm_path = (
                BASE_DIR /
                "confusion_matrix.png"
            )

            fig2.savefig(
                cm_path,
                dpi=200,
                bbox_inches="tight"
            )

            plt.close(fig2)

            # =====================================
            # INTERPRETASI CONFUSION MATRIX
            # =====================================

            tn, fp, fn, tp = matrix.ravel()

            if accuracy >= 0.90:
                kualitas = "sangat baik"

            elif accuracy >= 0.80:
                kualitas = "baik"

            elif accuracy >= 0.70:
                kualitas = "cukup baik"

            else:
                kualitas = "masih perlu ditingkatkan"

            with st.expander("Confusion Matrix", expanded=False):

                st.markdown(f"""
                    Sebanyak **{tn}** data **Service Ringan** berhasil diprediksi dengan benar sebagai **Service Ringan**, 
                    Sebanyak **{fp}** data **Service Ringan** diprediksi sebagai **Service Berat** sehingga termasuk kesalahan klasifikasi (*False Positive*), 
                    Sebanyak **{tp}** data **Service Berat** berhasil diprediksi dengan benar sebagai **Service Berat**, 
                    Sebanyak **{fn}** data **Service Berat** diprediksi sebagai **Service Ringan** sehingga termasuk kesalahan klasifikasi (*False Negative*).
                    """)

            # =====================================
            # FEATURE IMPORTANCE
            # =====================================

            # -------------------------------------
            # Membuat Grafik
            # -------------------------------------

            fig3, ax3 = plt.subplots(
                figsize=(4.8, 3.2),
                dpi=140
            )

            sns.barplot(
                data=importance_grouped,
                x="Importance",
                y="Fitur",
                palette="Reds_r",
                ax=ax3
            )

            ax3.set_xlabel(
                "Nilai Importance",
                fontsize=8
            )

            ax3.set_ylabel("")

            ax3.tick_params(
                labelsize=8
            )

            plt.tight_layout()

            # -------------------------------------
            # Grafik di tengah
            # -------------------------------------

            kiri, tengah, kanan = st.columns([2,3,2])

            with tengah:

                st.pyplot(
                    fig3,
                    use_container_width=False
                )

            # -------------------------------------
            # Simpan untuk PDF
            # -------------------------------------

            fi_path = (
                BASE_DIR /
                "feature_importance.png"
            )

            fig3.savefig(
                fi_path,
                dpi=250,
                bbox_inches="tight"
            )

            plt.close(fig3)

            # -------------------------------------
            # Data Top Feature
            # -------------------------------------

            top1 = importance_grouped.iloc[0]["Fitur"]
            top2 = importance_grouped.iloc[1]["Fitur"]
            top3 = importance_grouped.iloc[2]["Fitur"]

            # =====================================
            # PENJELASAN FEATURE IMPORTANCE
            # =====================================

            with st.expander("Feature Importance", expanded=False):

                ranking = importance_grouped.copy()

                ranking.insert(
                    0,
                    "No",
                    range(1, len(ranking) + 1)
                )

                ranking["Importance"] = ranking["Importance"].round(4)

                st.dataframe(
                    ranking,
                    hide_index=True,
                    use_container_width=True
                )

                st.markdown(f"""
                    Feature Importance menunjukkan tingkat kontribusi masing-masing fitur terhadap proses klasifikasi yang dilakukan oleh algoritma **Random Forest**.
                    Semakin besar nilai Feature Importance, semakin besar pula pengaruh suatu fitur dalam membantu model membedakan kategori **Service Ringan** dan **Service Berat**.
                    Berdasarkan hasil pelatihan model, fitur **{top1}** memiliki nilai Feature Importance tertinggi sehingga menjadi faktor utama dalam proses klasifikasi. Selanjutnya diikuti oleh fitur **{top2}** dan **{top3}** yang juga memberikan kontribusi penting terhadap keputusan model.
                """)

                 # =====================================
            # REPRESENTATIVE DECISION TREE
            # =====================================

            representative_tree = model.estimators_[0]

            fig_tree, ax_tree = plt.subplots(
                figsize=(18, 7),
                dpi=140
            )

            plot_tree(

                representative_tree,

                feature_names=feature_names,

                class_names=[
                    "Ringan",
                    "Berat"
                ],

                filled=True,

                rounded=True,

                fontsize=8,

                impurity=False,

                proportion=True,

                max_depth=3,

                ax=ax_tree

            )

            plt.tight_layout()

            kiri, tengah, kanan = st.columns([2,3,2])

            with tengah:

                st.pyplot(
                    fig_tree,
                    use_container_width=False
                )

            tree_path = (
                BASE_DIR /
                "representative_tree.png"
            )

            fig_tree.savefig(
                tree_path,
                dpi=250,
                bbox_inches="tight"
            )

            plt.close(fig_tree)

            # =====================================
            # REPRESENTATIVE DECISION PATH
            # =====================================

            tree_patterns = extract_tree_paths(
                representative_tree,
                feature_names
            )

            top_patterns = tree_patterns[:5]

            with st.expander(
                "Representative Decision Path",
                expanded=False
            ):

                st.caption(
                    "Lima pola keputusan dengan jumlah data terbesar pada Representative Decision Tree."
                )

                tabel = []

                for i, pattern in enumerate(top_patterns, start=1):

                    tabel.append({

                        "No": i,

                        "Pola Keputusan":
                            " → ".join(pattern["path"]),

                        "Prediksi":
                            pattern["prediction"],

                        "Jumlah Data":
                            pattern["samples"],

                        "Kemurnian":
                            f"{pattern['purity']:.1f}%"

                    })

                st.dataframe(
                    pd.DataFrame(tabel),
                    hide_index=True,
                    use_container_width=True
                )

                st.markdown("---")

                # =====================================
                # INSIGHT
                # =====================================

                best_pattern = top_patterns[0]

                characteristics = []

                for rule in best_pattern["path"]:

                    # -----------------------------
                    # Jenis Motor
                    # -----------------------------
                    if rule.startswith("Jenis_"):

                        if "> 0.50" in rule:

                            nama = (
                                rule.split("_")[1]
                                .split()[0]
                            )

                            characteristics.append(
                                f"Jenis motor **{nama}**"
                            )

                    # -----------------------------
                    # Indikasi
                    # -----------------------------
                    elif rule.startswith("Indikasi_"):

                        if "> 0.50" in rule:

                            nama = (
                                rule.split("_")[1]
                                .split()[0]
                            )

                            characteristics.append(
                                f"Indikasi kerusakan pada **{nama}**"
                            )

                    # -----------------------------
                    # Kilometer
                    # -----------------------------
                    elif rule.startswith("Km"):

                        if ">" in rule:

                            nilai = float(
                                rule.split(">")[1]
                            )

                            characteristics.append(
                                f"Kilometer kendaraan lebih dari **{nilai:,.0f} km**".replace(",", ".")
                            )

                        elif "≤" in rule:

                            nilai = float(
                                rule.split("≤")[1]
                            )

                            characteristics.append(
                                f"Kilometer kendaraan maksimal **{nilai:,.0f} km**".replace(",", ".")
                            )

                    # -----------------------------
                    # Usia Motor
                    # -----------------------------
                    elif rule.startswith("Usia Motor"):

                        if ">" in rule:

                            nilai = float(
                                rule.split(">")[1]
                            )

                            characteristics.append(
                                f"Usia motor lebih dari **{nilai:.0f} tahun**"
                            )

                        elif "≤" in rule:

                            nilai = float(
                                rule.split("≤")[1]
                            )

                            characteristics.append(
                                f"Usia motor maksimal **{nilai:.0f} tahun**"
                            )

                # Hilangkan duplikasi
                characteristics = list(
                    dict.fromkeys(characteristics)
                )

                st.markdown(
                    "#### Insight Representative Decision Path"
                )

                st.write(
                    """
Representative Decision Path merupakan salah satu jalur keputusan yang paling sering dilalui oleh data pada Representative Decision Tree. Jalur ini memberikan gambaran mengenai karakteristik kendaraan yang dominan pada data pelatihan.
                    """
                )

                if characteristics:

                    st.markdown(
                        "**Karakteristik kendaraan yang paling sering ditemukan:**"
                    )

                    for item in characteristics:

                        st.markdown(
                            f"- {item}"
                        )

                else:

                    st.info(
                        "Tidak ditemukan karakteristik khusus pada jalur keputusan ini."
                    )

                st.success(
                    f"""
Model memprediksi kendaraan pada pola tersebut termasuk kategori **{best_pattern['prediction']}**.

Pola ini muncul pada **{best_pattern['samples']}** data pelatihan dengan tingkat kemurnian sebesar **{best_pattern['purity']:.1f}%**.

Representative Decision Path memberikan gambaran mengenai pola kendaraan yang paling sering ditemukan pada data riwayat service sehingga dapat digunakan sebagai referensi dalam memahami karakteristik kendaraan yang cenderung memperoleh jenis layanan tertentu.
                    """
                )

            # =====================================
            # KARAKTERISTIK HASIL KLASIFIKASI
            # =====================================

            with st.expander(
                "Hasil Klasifikasi",
                expanded=False
            ):

                st.caption(
                    "Ringkasan karakteristik kendaraan berdasarkan hasil prediksi Random Forest."
                )

                urutan_jenis = [
                    "MAXi",
                    "Classy",
                    "Matic",
                    "Moped",
                    "Sport",
                    "Off-road",
                    "Unknown"
                ]

                for jenis in urutan_jenis:

                    data = summary_df[
                        summary_df["Jenis"] == jenis
                    ].copy()

                    if data.empty:
                        continue

                    st.info(f"{jenis}")

                    tampil = data[
                        [
                            "Service",
                            "Rata-rata KM",
                            "Rata-rata Usia"
                        ]
                    ].copy()

                    tampil.columns = [
                        "Hasil Klasifikasi",
                        "Rata-rata KM (km)",
                        "Rata-rata Usia (Tahun)"
                    ]

                    tampil["Rata-rata KM (km)"] = (
                        tampil["Rata-rata KM (km)"]
                        .round(0)
                        .astype(int)
                    )

                    tampil["Rata-rata Usia (Tahun)"] = (
                        tampil["Rata-rata Usia (Tahun)"]
                        .round(1)
                    )

                    st.dataframe(
                        tampil,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ==================================
                    # PENJELASAN OTOMATIS
                    # ==================================

                    ringan = data[
                        data["Service"] == "Ringan"
                    ]

                    berat = data[
                        data["Service"] == "Berat"
                    ]

                    if not ringan.empty and not berat.empty:

                        km_ringan = ringan.iloc[0]["Rata-rata KM"]
                        usia_ringan = ringan.iloc[0]["Rata-rata Usia"]

                        km_berat = berat.iloc[0]["Rata-rata KM"]
                        usia_berat = berat.iloc[0]["Rata-rata Usia"]

                        st.success(f"""
Berdasarkan hasil klasifikasi Random Forest, kendaraan jenis **{jenis}** yang diprediksi sebagai **Service Ringan** memiliki rata-rata kilometer sekitar **{km_ringan:,.0f} km** dan rata-rata usia motor sekitar **{usia_ringan:.1f} tahun**.
Sedangkan kendaraan yang diprediksi sebagai **Service Berat** memiliki rata-rata kilometer sekitar **{km_berat:,.0f} km** dan rata-rata usia motor sekitar **{usia_berat:.1f} tahun**.

Hal ini menunjukkan bahwa pada kendaraan jenis **{jenis}**, kendaraan dengan kilometer dan usia motor yang lebih tinggi cenderung diklasifikasikan sebagai **Service Berat**.
                        """)

                    elif not ringan.empty:

                        km_ringan = ringan.iloc[0]["Rata-rata KM"]
                        usia_ringan = ringan.iloc[0]["Rata-rata Usia"]

                        st.success(f"""
Berdasarkan hasil klasifikasi Random Forest, seluruh kendaraan jenis **{jenis}** pada data ini diprediksi sebagai **Service Ringan**.
Rata-rata kendaraan memiliki kilometer sekitar **{km_ringan:,.0f} km** dengan usia motor sekitar **{usia_ringan:.1f} tahun**.

Pada data yang digunakan belum ditemukan kendaraan jenis **{jenis}** yang diprediksi sebagai **Service Berat**.
                        """)

                    elif not berat.empty:

                        km_berat = berat.iloc[0]["Rata-rata KM"]
                        usia_berat = berat.iloc[0]["Rata-rata Usia"]

                        st.success(f"""
Berdasarkan hasil klasifikasi Random Forest, seluruh kendaraan jenis **{jenis}** pada data ini diprediksi sebagai **Service Berat**.
Rata-rata kendaraan memiliki kilometer sekitar **{km_berat:,.0f} km** dengan usia motor sekitar **{usia_berat:.1f} tahun**.

Pada data yang digunakan belum ditemukan kendaraan jenis **{jenis}** yang diprediksi sebagai **Service Ringan**.
                        """)

                    st.markdown("<br>", unsafe_allow_html=True)

            # =====================================
            # LAPORAN PDF
            # =====================================

            st.markdown("---")

            logo_path = BASE_DIR / "assets" / "yamaha_logo.png"

            pdf_path = generate_pdf(

            pdf_path=BASE_DIR / "laporan_training_model.pdf",
                
                logo_path=logo_path,
                nama_file=uploaded_file.name,
                total_data=len(df),
                duplicate_data=total_duplicate,
                missing_value=total_missing,
                train_data=train_size,
                test_data=test_size,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                distribution_image=distribution_path,
                cm_image=cm_path,
                fi_image=fi_path,
                top_features=(
                    importance_grouped[
                        ["Fitur", "Importance"]
                    ]
                    .head(5)
                    .values
                    .tolist()
                ),
                summary_df=summary_df,
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp
            )

            left, center, right = st.columns([1, 2, 1])

            with center:

                with open(pdf_path, "rb") as pdf_file:

                    st.download_button(
                        label="📄 Download Laporan PDF",
                        data=pdf_file.read(),
                        file_name="Laporan_Training_Model.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf",
                        on_click="ignore"
                    )

            st.markdown(
                """
                <div style="
                    text-align:center;
                    color:#9ca3af;
                    font-size:14px;
                    margin-top:-8px;
                    margin-bottom:15px;
                ">
                    Tekan tombol di atas untuk mengunduh laporan hasil pelatihan model dalam format PDF.
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:

        st.exception(e)
