from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import (
    TA_LEFT,
    TA_CENTER,
    TA_RIGHT
)
from reportlab.platypus import (
    HRFlowable
)
from reportlab.lib.units import cm
from datetime import datetime
from pathlib import Path
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import KeepTogether

import os


# ==========================================================
# GENERATE PDF
# ==========================================================

def generate_pdf(
    pdf_path,
    logo_path,
    nama_file,
    total_data,
    duplicate_data,
    missing_value,
    train_data,
    test_data,
    accuracy,
    precision,
    recall,
    f1,
    distribution_image,
    cm_image,
    fi_image,
    top_features,
    summary_df,
    tn,
    fp,
    fn,
    tp
):
    
    # =====================================
    # VALIDASI PATH
    # =====================================

    pdf_path = str(Path(pdf_path))
    logo_path = str(Path(logo_path))

    distribution_image = str(Path(distribution_image))

    cm_image = str(Path(cm_image))
    fi_image = str(Path(fi_image))

    # =====================================
    # MEMBUAT DOKUMEN
    # =====================================

    doc = SimpleDocTemplate(
        pdf_path,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1 * cm,
        bottomMargin=1 * cm
    )

    styles = getSampleStyleSheet()

    cm_style = ParagraphStyle(
        "CMStyle",
        parent=styles["BodyText"],
        alignment=TA_JUSTIFY,   # Rata kanan-kiri
        leading=18,             # Jarak antar baris
        leftIndent=8,
        rightIndent=5,
        spaceBefore=0,
        spaceAfter=0,
    )

    elements = []

    # ==========================================================
    # HEADER
    # ==========================================================

    company_style = ParagraphStyle(
        "CompanyStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=22,
        alignment=TA_LEFT,
        textColor=colors.black
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#555555")
    )

    contact_style = ParagraphStyle(
        "ContactStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#666666")
    )

    report_title_style = ParagraphStyle(
        "ReportTitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        alignment=TA_CENTER
    )

    # ==========================================================
    # LOGO
    # ==========================================================

    if os.path.exists(logo_path):

        logo = Image(
            logo_path,
            width=3.8 * cm,
            height=2.5 * cm
        )
    else:
        logo = Spacer(3.8 * cm, 2.5 * cm)

    # ==========================================================
    # INFORMASI PERUSAHAAN
    # ==========================================================

    tanggal = datetime.now().strftime("%d %B %Y %H:%M:%S")

    company_info = Table(
        [
            [
                Paragraph(
                    "PT. YAMAHA INDONESIA MOTOR MANUFACTURING",
                    company_style
                )
            ],
            [
                Paragraph(
                    "YAMAHA TJAHAJA BARU TABING",
                    subtitle_style
                )
            ],
            [
                Paragraph(
                    "Jl. Prof. Dr. Hamka No.56, Parupuk Tabing, Kec. Koto Tangah, Kota Padang, Sumatera Barat 25173",
                    contact_style
                )
            ],
            [
                Paragraph(
                    "08116606631  |  https://tjahaja-baru.com/",
                    contact_style
                )
            ],
            [
                Paragraph(
                    f"{tanggal}",
                    contact_style
                )
            ],
        ],
        colWidths=[13.5 * cm]
    )

    company_info.setStyle(
        TableStyle([
            ("LEFTPADDING",(0,0),(-1,-1),0),
            ("RIGHTPADDING",(0,0),(-1,-1),0),
            ("TOPPADDING",(0,0),(-1,-1),0),
            ("BOTTOMPADDING",(0,0),(-1,-1),1),
            ("VALIGN",(0,0),(-1,-1),"TOP")
        ])
    )

    # ==========================================================
    # HEADER TABLE
    # ==========================================================

    header_table = Table(
        [
            [
                logo,
                company_info
            ]
        ],
        colWidths=[4.2 * cm, 13.8 * cm]
    )

    header_table.setStyle(
        TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),0),
            ("RIGHTPADDING",(0,0),(-1,-1),0),
            ("TOPPADDING",(0,0),(-1,-1),0),
            ("BOTTOMPADDING",(0,0),(-1,-1),0)
        ])
    )

    elements.append(header_table)

    elements.append(
        Spacer(1, 8)
    )

    # ==========================================================
    # GARIS PEMBATAS
    # ==========================================================

    elements.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#C00000")
        )
    )

    elements.append(
        Spacer(1, 8)
    )

    # =====================================
    # PENJELASAN LAPORAN
    # =====================================

    elements.append(
        Spacer(1, 8)
    )

    penjelasan = """
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    Laporan ini menyajikan informasi hasil klasifikasi layanan service
    kendaraan yamaha yang dihasilkan oleh sistem menggunakan algoritma
    random forest. Berikut informasi mengenai dataset yang digunakan
    pada klasifikasi layanan service kendaraan.
    """

    elements.append(
        Paragraph(
            penjelasan,
            cm_style
        )
    )

    # =====================================
    # INFORMASI DATASET
    # =====================================
    
    elements.append(
        Spacer(1, 5)
    )

    dataset_table = Table(
        [
            ["Nama File", ":", nama_file],
            ["Jumlah Data", ":", f"{total_data} data"],
            ["Data Duplikat", ":", f"{duplicate_data} data"],
            ["Missing Value", ":", f"{missing_value} data"],
            ["Distribusi Target", ":",f""],
        ],
        colWidths=[165, 10, 305]
    )

    dataset_table.setStyle(
        TableStyle([
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING", (0,0), (-1,-1), 18),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("VALIGN", (0,0), (-1,-1), "TOP"),

            # Tidak ada garis sehingga tampil seperti dokumen Word
            ("LINEBELOW", (0,0), (-1,-1), 0, colors.white),
            ("LINEABOVE", (0,0), (-1,-1), 0, colors.white),
            ("LINEBEFORE", (0,0), (-1,-1), 0, colors.white),
            ("LINEAFTER", (0,0), (-1,-1), 0, colors.white),
        ])
    )

    elements.append(dataset_table)
    
    elements.append(
        Spacer(1, 5)
    )

    # =====================================
    # DISTRIBUSI TARGET
    # =====================================

    distribution_img = Image(
        distribution_image,
        width=470,
        height=210
    )

    distribution_table = Table(
        [[distribution_img]],
        colWidths=[470]
    )

    distribution_table.setStyle(
        TableStyle([
            ("BOX",(0,0),(-1,-1),0.8,colors.black),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), 
            ("LEFTPADDING", (0, 0), (-1, -1), 0), 
            ("RIGHTPADDING", (0, 0), (-1, -1), 10), 
            ("TOPPADDING", (0, 0), (-1, -1), 0), 
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )

    elements.append(distribution_table)

    elements.append(
        Spacer(1,10)
    )

    # ==========================================================
    # GARIS PEMBATAS
    # ==========================================================

    elements.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#C00000")
        )
    )

    elements.append(
        Spacer(1, 10)
    )

    # =====================================
    # HASIL EVALUASI MODEL
    # =====================================

    hasil_evaluasi = f"""
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    Model Random Forest dievaluasi menggunakan
    {train_data} data training dan
    {test_data} data testing. Berdasarkan hasil pengujian,
    model memperoleh Accuracy sebesar {accuracy:.2%},
    Precision {precision:.2%},
    Recall {recall:.2%}, dan
    F1-Score {f1:.2%}. Nilai tersebut menunjukkan bahwa model
    memiliki kemampuan yang baik.
    """

    elements.append(
        Paragraph(
            hasil_evaluasi,
            cm_style
        )
    )

    elements.append(
        Spacer(1, 5)
    )

    # ==========================================================
    # GARIS PEMBATAS
    # ==========================================================

    elements.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#666666")
        )
    )

    elements.append(
        Spacer(1, 5)
    )

    # =====================================
    # CONFUSION MATRIX
    # =====================================
    
    cm_img = Image(cm_image, width=180, height=180)

    cm_style = ParagraphStyle(
        "CMStyle",
        parent=styles["BodyText"],
        alignment=TA_JUSTIFY,   # Rata kanan-kiri
        leading=18,             # Jarak antar baris
        leftIndent=8,
        rightIndent=5,
        spaceBefore=0,
        spaceAfter=0,
    )

    elements.append(
        Spacer(1, 5)
    )
    # Penjelasan
    cm_desc = Paragraph(
        f"""
        Sebanyak {tn} data Service Ringan berhasil diprediksi dengan benar.Sementara itu, terdapat {fp} data Service Ringan yang diprediksi sebagai Service Berat.
        Lalu sebanyak {tp} data Service Berat berhasil diprediksi dengan benar,sedangkan {fn} data diprediksi sebagai Service Ringan.
        """,
        cm_style
    )

    # Tabel 2 kolom tanpa border
    cm_table = Table(
        [[cm_img, cm_desc]],
        colWidths=[190, 290]
    )

    cm_table.setStyle(TableStyle([ 
        ("VALIGN", (0, 0), (-1, -1), "TOP"), 
        ("LEFTPADDING", (0, 0), (-1, -1), 0), 
        ("RIGHTPADDING", (0, 0), (-1, -1), 10), 
        ("TOPPADDING", (0, 0), (-1, -1), 0), 
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0), 
    ]))

    elements.append(cm_table)

    elements.append(
        Spacer(1,5)
    )
    
    # =====================================
    # FEATURE IMPORTANCE
    # =====================================
    # ------------------------------------
    # Gambar
    # ------------------------------------

    fi_img = Image(
        fi_image,
        width=300,
        height=200
    )

    fi_table = Table(
        [[fi_img]],
        colWidths=[470]
    )

    fi_table.setStyle(TableStyle([

        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))

    elements.append(fi_table)

    elements.append(
        Spacer(1,5)
    )

    # ------------------------------------
    # Penjelasan
    # ------------------------------------

    elements.append(
        Paragraph(
            """
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Feature Importance menunjukkan tingkat kontribusi setiap fitur
            terhadap proses klasifikasi layanan service menggunakan algoritma
            Random Forest. Semakin tinggi nilai importance suatu fitur,
            semakin besar pengaruhnya dalam membantu model menentukan hasil
            klasifikasi layanan service kendaraan.
            """,
            cm_style
        )
    )

    elements.append(
        Spacer(1,5)
    )

    # ------------------------------------
    # Tabel Ranking
    # ------------------------------------

    feature_data = [
        ["No","Nama Fitur","Importance"]
    ]

    for i, (fitur, nilai) in enumerate(top_features, start=1):

        feature_data.append([
            str(i),
            fitur,
            f"{nilai:.4f}"
        ])

    feature_table = Table(
        feature_data,
        colWidths=[45,250,100]
    )

    feature_table.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("BOX",(0,0),(-1,-1),1,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E5E7EB")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))

    elements.append(feature_table)

    elements.append(
        Spacer(1,10)
    )

    # ------------------------------------
    # Interpretasi
    # ------------------------------------

    nama_fitur = [x[0] for x in top_features]

    elements.append(
        Paragraph(
            f"""
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Berdasarkan hasil Feature Importance, fitur
            {nama_fitur[0]} memiliki nilai importance terbesar,
            sehingga memberikan kontribusi paling dominan dalam proses
            klasifikasi layanan service. Selanjutnya diikuti oleh
            {', '.join(nama_fitur[1:])} yang juga memberikan
            pengaruh terhadap keputusan model Random Forest dalam
            mengklasifikasikan layanan Service Ringan maupun Service Berat.
            """,
            cm_style
        )
    )

    elements.append(
        Spacer(1,5)
    )

    # ==========================================================
    # GARIS PEMBATAS
    # ==========================================================

    elements.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#C00000")
        )
    )

    elements.append(
        Spacer(1, 8)
    )

    # =====================================
    # HASIL KLASIFIKASI
    # =====================================

    elements.append(
        Paragraph(
            """
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Hasil klasifikasi berikut menunjukkan karakteristik kendaraan
            berdasarkan hasil prediksi algoritma Random Forest. Setiap jenis
            kendaraan disajikan seperti berikut :
            """,
            cm_style
        )
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

        blok = []

        # =====================================
        # JUDUL
        # =====================================

        blok.append(
            Paragraph(
                f"<b>{jenis}</b>",
                styles["Heading3"]
            )
        )

        blok.append(Spacer(1, 5))

        # =====================================
        # TABEL
        # =====================================

        table_data = [[
            "Hasil Klasifikasi",
            "Rata-rata KM (km)",
            "Rata-rata Usia (Tahun)"
        ]]

        for _, row in data.iterrows():

            table_data.append([
                row["Service"],
                f"{row['Rata-rata KM']:,.0f}".replace(",", "."),
                f"{row['Rata-rata Usia']:.1f}"
            ])

        hasil_table = Table(
            table_data,
            colWidths=[150, 160, 160],
            repeatRows=1
        )

        hasil_table.setStyle(TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E5E7EB")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("BOX",(0,0),(-1,-1),0.8,colors.grey),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6)

        ]))

        blok.append(hasil_table)

        blok.append(Spacer(1, 5))

        # =====================================
        # INTERPRETASI
        # =====================================

        ringan = data[data["Service"] == "Ringan"]
        berat = data[data["Service"] == "Berat"]

        if not ringan.empty and not berat.empty:

            km_ringan = ringan.iloc[0]["Rata-rata KM"]
            usia_ringan = ringan.iloc[0]["Rata-rata Usia"]

            km_berat = berat.iloc[0]["Rata-rata KM"]
            usia_berat = berat.iloc[0]["Rata-rata Usia"]

            interpretasi = f"""
            Berdasarkan hasil klasifikasi Random Forest, kendaraan jenis
            <b>{jenis}</b> yang diprediksi sebagai <b>Service Ringan</b>
            memiliki rata-rata kilometer sekitar <b>{km_ringan:,.0f} km</b>
            dengan rata-rata usia <b>{usia_ringan:.1f} tahun</b>.
            Sementara itu kendaraan yang diprediksi sebagai
            <b>Service Berat</b> memiliki rata-rata kilometer sekitar
            <b>{km_berat:,.0f} km</b> dengan rata-rata usia
            <b>{usia_berat:.1f} tahun</b>. Hal ini menunjukkan bahwa
            kendaraan dengan usia dan kilometer yang lebih tinggi
            cenderung memperoleh layanan <b>Service Berat</b>.
            """

        elif not ringan.empty:

            km_ringan = ringan.iloc[0]["Rata-rata KM"]
            usia_ringan = ringan.iloc[0]["Rata-rata Usia"]

            interpretasi = f"""
            Berdasarkan hasil klasifikasi Random Forest, seluruh kendaraan
            jenis <b>{jenis}</b> pada data ini diprediksi sebagai
            <b>Service Ringan</b> dengan rata-rata kilometer sekitar
            <b>{km_ringan:,.0f} km</b> dan rata-rata usia
            <b>{usia_ringan:.1f} tahun</b>.
            """

        else:

            km_berat = berat.iloc[0]["Rata-rata KM"]
            usia_berat = berat.iloc[0]["Rata-rata Usia"]

            interpretasi = f"""
            Berdasarkan hasil klasifikasi Random Forest, seluruh kendaraan
            jenis <b>{jenis}</b> pada data ini diprediksi sebagai
            <b>Service Berat</b> dengan rata-rata kilometer sekitar
            <b>{km_berat:,.0f} km</b> dan rata-rata usia
            <b>{usia_berat:.1f} tahun</b>.
            """

        blok.append(
            Paragraph(
                interpretasi,
                cm_style
            )
        )

        blok.append(
            Spacer(1, 10)
        )

        # =====================================
        # SATUKAN AGAR TIDAK TERPISAH HALAMAN
        # =====================================

        elements.append(
            KeepTogether(blok)
        )

        # =====================================
        # GARIS PEMISAH
        # =====================================

        if jenis != urutan_jenis[-1]:

            elements.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#BDBDBD")
                )
            )

            elements.append(
                Spacer(1, 10)
            )
            
    # =====================================
    # KESIMPULAN
    # =====================================

    elements.append(

        Paragraph(

            "<b>KESIMPULAN</b>",

            styles["Heading3"]

        )

    )

    elements.append(
        Spacer(1,5)
    )

    kesimpulan = f"""
    Model Random Forest berhasil dilatih menggunakan dataset
    layanan service kendaraan Yamaha.

    Berdasarkan proses pelatihan diperoleh nilai Accuracy sebesar
    <b>{accuracy:.2%}</b>, Precision sebesar
    <b>{precision:.2%}</b>, Recall sebesar
    <b>{recall:.2%}</b>, dan F1-Score sebesar
    <b>{f1:.2%}</b>.

    Hasil evaluasi menunjukkan bahwa model mampu melakukan
    klasifikasi layanan <b>Service Ringan</b> dan
    <b>Service Berat</b> dengan performa yang baik.

    Feature Importance menunjukkan bahwa beberapa fitur utama
    memiliki kontribusi paling besar terhadap proses klasifikasi,
    sehingga dapat digunakan sebagai dasar dalam pengambilan
    keputusan layanan service kendaraan.
    """

    kesimpulan_table = Table(

        [[

            Paragraph(

                kesimpulan,

                styles["BodyText"]

            )

        ]],

        colWidths=[470]

    )

    kesimpulan_table.setStyle(

        TableStyle([

            ("BOX",(0,0),(-1,-1),1,colors.black),

            ("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),

            ("BOTTOMPADDING",(0,0),(-1,-1),10),

            ("TOPPADDING",(0,0),(-1,-1),10)

        ])

    )

    elements.append(kesimpulan_table)

    elements.append(
        Spacer(1,10)
    )

    # =====================================
    # PENUTUP LAPORAN
    # =====================================

    elements.append(
        Paragraph(
            "<para align='center'><font size='9' color='grey'>"
            "Laporan ini dibuat secara otomatis oleh Sistem "
            "Klasifikasi Layanan Service Kendaraan Yamaha "
            "menggunakan algoritma Random Forest."
            "</font></para>",
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(1, 8)
    )

    # =====================================
    # MEMBUAT PDF
    # =====================================

    try:

        doc.build(elements)

    except Exception as e:

        raise Exception(
            f"Gagal membuat laporan PDF: {str(e)}"
        )

    # =====================================
    # RETURN PATH PDF
    # =====================================

    return pdf_path
