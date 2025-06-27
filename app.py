import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import joblib


scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="DBSCAN Anomaly Detector", layout="wide")
st.title("💥 DBSCAN-based Anomaly Detection App")

st.markdown("""
Upload your CSV file.
This app will:
- Scale your data
- Cluster it using DBSCAN (live)
- Visualize results
""")

# File uploader
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    for col in ["Class", "Time"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)


    st.subheader("📊 Dataset Preview")
    st.write(df.head())
    st.write("📋 Columns:", df.columns.tolist())

    try:
        df = df.select_dtypes(include=[np.number])
        df = df.dropna()

        scaled_data = scaler.transform(df)

        dbscan = DBSCAN(eps=1.5, min_samples=5)
        labels = dbscan.fit_predict(scaled_data)

        df["Cluster"] = labels

        st.subheader("📌 Cluster Distribution")
        cluster_counts = df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        st.dataframe(cluster_counts)

        outliers = (labels == -1).sum()
        total = len(labels)
        st.markdown(f"**Anomalies Detected:** `{outliers}` out of `{total}` rows ({(outliers/total)*100:.2f}%)")

        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        df_pca = pd.DataFrame(components, columns=["PCA1", "PCA2"])
        df_pca["Cluster"] = labels

        st.subheader("📉 PCA Cluster Plot (2D Projection)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=40, legend=False)
        plt.title("DBSCAN Clusters on PCA-Reduced Data")
        st.pyplot(fig)

        st.subheader("📥 Download Results")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="dbscan_clustered_output.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ Error during processing.")
        st.exception(e)

with st.expander("📸 See Sample Cluster Plot"):
    st.image("dbscan_sample_clusters.png", caption="DBSCAN Cluster Sample", use_container_width=True)
