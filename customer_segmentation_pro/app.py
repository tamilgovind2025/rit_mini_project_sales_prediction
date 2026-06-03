import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.title("Customer Segmentation & Recommendation System")

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("Mall_Customers.csv")

# Rename columns for simplicity

df.columns = [
    "CustomerID",
    "Gender",
    "Age",
    "AnnualIncome",
    "SpendingScore"
]

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.header("Dashboard Controls")

num_clusters = st.sidebar.slider(
    "Select Number of Clusters",
    2,
    10,
    5
)

# -----------------------------------
# Display Dataset
# -----------------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head())

# -----------------------------------
# Charts
# -----------------------------------

st.subheader("Customer Analysis Charts")

col1, col2 = st.columns(2)

with col1:

    fig_income = px.histogram(
        df,
        x="AnnualIncome",
        nbins=20,
        title="Annual Income Distribution"
    )

    st.plotly_chart(fig_income)

with col2:

    fig_spending = px.histogram(
        df,
        x="SpendingScore",
        nbins=20,
        title="Spending Score Distribution"
    )

    st.plotly_chart(fig_spending)

# -----------------------------------
# Gender Distribution
# -----------------------------------

fig_gender = px.pie(
    df,
    names="Gender",
    title="Gender Distribution"
)

st.plotly_chart(fig_gender)

# -----------------------------------
# K-Means Clustering
# -----------------------------------

st.subheader("K-Means Customer Segmentation")

X = df[["AnnualIncome", "SpendingScore"]]

# Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# KMeans

kmeans = KMeans(
    n_clusters=num_clusters,
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------------
# Cluster Visualization
# -----------------------------------

fig_cluster = px.scatter(
    df,
    x="AnnualIncome",
    y="SpendingScore",
    color=df["Cluster"].astype(str),
    title="Customer Segments",
    hover_data=["CustomerID", "Age"]
)

st.plotly_chart(fig_cluster, use_container_width=True)

# -----------------------------------
# Cluster Summary
# -----------------------------------

st.subheader("Cluster Summary")

cluster_summary = df.groupby("Cluster").mean(
    numeric_only=True
)

st.dataframe(cluster_summary)

# -----------------------------------
# Recommendation System
# -----------------------------------

st.subheader("Customer Recommendation Engine")

customer_id = st.number_input(
    "Enter Customer ID",
    min_value=1,
    max_value=int(df["CustomerID"].max()),
    value=1
)

customer_data = df[df["CustomerID"] == customer_id]

if len(customer_data) > 0:

    cluster = customer_data["Cluster"].values[0]

    st.success(f"Customer belongs to Cluster {cluster}")

    # Recommendations based on cluster

    if cluster == 0:
        st.info(
            "Recommended:\n"
            "- Premium Products\n"
            "- Luxury Membership\n"
            "- High-End Fashion"
        )

    elif cluster == 1:
        st.info(
            "Recommended:\n"
            "- Budget Offers\n"
            "- Discount Coupons\n"
            "- Seasonal Sales"
        )

    elif cluster == 2:
        st.info(
            "Recommended:\n"
            "- Electronics\n"
            "- Trending Products\n"
            "- Combo Offers"
        )

    elif cluster == 3:
        st.info(
            "Recommended:\n"
            "- Travel Packages\n"
            "- Lifestyle Products\n"
            "- Fitness Membership"
        )

    else:
        st.info(
            "Recommended:\n"
            "- Personalized Products\n"
            "- Gift Cards\n"
            "- Loyalty Rewards"
        )

# -----------------------------------
# Download Processed Dataset
# -----------------------------------

st.subheader("Download Clustered Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "clustered_customers.csv",
    "text/csv"
)

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.markdown(
    "Built with Streamlit + K-Means Clustering"
)