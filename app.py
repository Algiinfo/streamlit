# =========================
# 1. IMPORT LIBRARY
# =========================
import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 2. KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="PT Wira Lodya")

st.title("Sistem Pompa Industri")
st.subheader("PT Wira Lodya Utama")


# =========================
# 3. LOAD DATASET (FIXED)
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset_PT_WLU.csv")

# Debug (boleh dihapus nanti)
st.write("Isi folder data:", os.listdir(os.path.join(BASE_DIR, "data")))

df = pd.read_csv(DATA_PATH, sep=";")

st.success("Dataset berhasil dimuat ✅")
st.dataframe(df.head())


# =========================
# 4. PREPROCESSING DATA
# =========================
st.subheader("Preprocessing Data")

# Pastikan kolom numerik
numeric_cols = ["capacity_m3h", "head_m", "power_kw", "price_idr"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Drop data tidak lengkap
df = df.dropna(subset=numeric_cols)

# Hapus duplikasi produk (jika ada)
if "product_id" in df.columns:
    df = df.drop_duplicates(subset="product_id")
else:
    df["product_id"] = df.index.astype(str)

st.success("Preprocessing data selesai")
st.write("Jumlah produk:", df.shape[0])

st.subheader("Sample Data Produk (Clean)")
st.dataframe(df.head())


# =========================
# 5. DATA TRANSFORMATION
# =========================
st.subheader("Data Transformation (Normalisasi)")

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[numeric_cols])

df_scaled = pd.DataFrame(
    scaled_features,
    columns=numeric_cols,
    index=df["product_id"]
)

st.dataframe(df_scaled.head())


# =========================
# 6. CONTENT-BASED FILTERING
# =========================
def recommend_similar_products(product_id, feature_matrix, products_df, top_n=3):
    idx = feature_matrix.index.get_loc(product_id)
    similarity = cosine_similarity(feature_matrix)
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n + 1]

    product_indices = [i[0] for i in scores]
    recommended_ids = feature_matrix.index[product_indices]

    return products_df[products_df["product_id"].isin(recommended_ids)]


# =========================
# 7. INPUT USER
# =========================
st.header("Input Kebutuhan Pelanggan")

capacity = st.number_input("Capacity minimum (m3/jam)", min_value=0.0)
head = st.number_input("Head minimum (meter)", min_value=0.0)


# =========================
# 8. REKOMENDASI
# =========================
st.subheader("Rekomendasi Pompa")

if st.button("Rekomendasikan Pompa"):
    st.info("Menjalankan sistem rekomendasi...")

    filtered_df = df[
        (df["capacity_m3h"] >= capacity) &
        (df["head_m"] >= head)
    ]

    if filtered_df.empty:
        st.warning("Tidak ada produk yang sesuai kriteria")
    else:
        sample_id = filtered_df["product_id"].iloc[0]

        recommendations = recommend_similar_products(
            sample_id,
            df_scaled,
            df
        )

        st.dataframe(
            recommendations[
                ["product_name", "capacity_m3h", "head_m", "power_kw", "price_idr"]
            ]
        )
