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
# 3. LOAD DATASET
# =========================
BASE_DIR = os.path.dirname(__file__)

df_products = pd.read_csv(os.path.join(BASE_DIR, "data", "products.csv"))
df_customers = pd.read_csv(os.path.join(BASE_DIR, "data", "customers.csv"))
df_transactions = pd.read_csv(os.path.join(BASE_DIR, "data", "transactions.csv"))
df_transaction_details = pd.read_csv(
    os.path.join(BASE_DIR, "data", "transaction_details.csv")
)

st.success("Dataset berhasil dimuat ✅")


# =========================
# 4. PREPROCESSING DATA
# =========================
st.subheader("Preprocessing Data")

# Hapus duplikasi
df_products = df_products.drop_duplicates(subset="product_id")
df_customers = df_customers.drop_duplicates(subset="customer_id")
df_transactions = df_transactions.drop_duplicates(subset="transaction_id")

# Filter transaksi valid
df_transactions = df_transactions[df_transactions["status"] == "Completed"]

# Hapus data produk tidak lengkap
df_products = df_products.dropna(subset=[
    "capacity_m3h",
    "head_m",
    "power_kw",
    "price_idr"
])

st.success("Preprocessing data selesai")

st.write("Jumlah produk setelah preprocessing:", df_products.shape[0])
st.write("Jumlah transaksi valid:", df_transactions.shape[0])

st.subheader("Sample Data Produk (Clean)")
st.dataframe(df_products.head())


# =========================
# 5. DATA TRANSFORMATION
# =========================
st.subheader("Data Transformation (Normalisasi)")

numeric_features = df_products[
    ["capacity_m3h", "head_m", "power_kw", "price_idr"]
]

scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

df_numeric_scaled = pd.DataFrame(
    numeric_scaled,
    columns=numeric_features.columns,
    index=df_products["product_id"]
)

st.write("Fitur numerik setelah normalisasi:")
st.dataframe(df_numeric_scaled.head())


# =========================
# 6. FUNGSI CONTENT-BASED FILTERING
# =========================
def recommend_similar_products(product_id, feature_matrix, products_df, top_n=3):
    idx = feature_matrix.index.get_loc(product_id)

    similarity_scores = cosine_similarity(feature_matrix)
    sim_scores = list(enumerate(similarity_scores[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]
    recommended_ids = feature_matrix.index[product_indices]

    return products_df[products_df["product_id"].isin(recommended_ids)]


# =========================
# 7. INPUT KEBUTUHAN PELANGGAN
# =========================
st.header("Input Kebutuhan Pelanggan")

industry = st.selectbox(
    "Industri Pelanggan",
    df_customers["industry"].dropna().unique()
)

capacity = st.number_input("Capacity (m3/jam)", min_value=0.0)
head = st.number_input("Head Pompa (meter)", min_value=0.0)

material = st.selectbox(
    "Material Pompa",
    df_products["material"].dropna().unique()
)


# =========================
# 8. CONTENT-BASED FILTERING (REKOMENDASI)
# =========================
st.subheader("Content-Based Filtering")

if st.button("Rekomendasikan Pompa"):
    st.info("Menjalankan Content-Based Filtering...")

    sample_product_id = df_products["product_id"].iloc[0]

    recommendations = recommend_similar_products(
        sample_product_id,
        df_numeric_scaled,
        df_products
    )

    st.subheader("Rekomendasi Produk Pompa")
    st.dataframe(recommendations[
        ["product_name", "capacity_m3h", "head_m", "power_kw", "price_idr"]
    ])
