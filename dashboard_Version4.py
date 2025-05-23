import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

from eda_utils import (
    clean_data, get_summary_stats, get_missing_values, top_n_counts,
    add_features, get_outliers, filter_for_demo
)

st.set_page_config(page_title="Product Recommendation System", layout="wide")
st.title("📦 Product Recommendation System Dashboard")
st.markdown("This dashboard demonstrates EDA and product recommendations using the Amazon Electronics ratings dataset.")

# --- Data Upload ---
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose ratings_Electronics.csv", type="csv")
if not uploaded_file:
    st.info("Please upload the dataset to start.")
    st.stop()
df = pd.read_csv(uploaded_file, names=['userId', 'productId', 'rating', 'timestamp'])
df = clean_data(df)

# --- Data Cleaning & Integrity ---
st.header("1. Data Cleaning & Integrity Checks")
st.write("**Missing values per column:**")
st.write(get_missing_values(df))
st.write("**Shape:**", df.shape)
st.write("**Duplicate rows:**", df.duplicated().sum())
st.write("**Data types:**")
st.write(df.dtypes)

# --- Feature Engineering ---
st.header("2. Feature Engineering")
df = add_features(df)
st.dataframe(df.head(), use_container_width=True)

# --- Summary Statistics ---
st.header("3. Summary Statistics & Insights")
summary = get_summary_stats(df)
st.write("**Unique users:**", summary['n_unique_users'])
st.write("**Unique products:**", summary['n_unique_products'])
st.write("**Ratings:**")
st.write(summary['rating_stats'])
st.write("**Most active users:**")
st.write(top_n_counts(df, 'userId'))
st.write("**Most rated products:**")
st.write(top_n_counts(df, 'productId'))

# --- Patterns & Outliers ---
st.header("4. Patterns, Trends & Anomalies")
outliers = get_outliers(df)
st.write(f"**Number of outlier ratings:** {len(outliers)}")
if not outliers.empty:
    st.dataframe(outliers.head(10))

# --- Visualizations ---
st.header("5. Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['rating'], bins=5, kde=True, ax=ax)
    ax.set_xlabel("Rating")
    st.pyplot(fig)

with col2:
    st.subheader("Top 10 Most Rated Products")
    top_products = top_n_counts(df, 'productId')
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_products.values, y=top_products.index, ax=ax2)
    ax2.set_xlabel("Count")
    ax2.set_ylabel("ProductId")
    st.pyplot(fig2)

# --- Interactive Data Explorer ---
st.header("6. Data Explorer")
st.dataframe(df.sample(50), use_container_width=True)

# --- Recommendation System ---
st.header("7. Product Recommendation Demo (User-based Collaborative Filtering)")
df_demo = filter_for_demo(df)
pivot = df_demo.pivot_table(index='userId', columns='productId', values='rating').fillna(0)
user_ids = pivot.index.tolist()
user_input = st.selectbox("Select a user for recommendations", user_ids)

if user_input:
    user_sim = cosine_similarity(pivot)
    user_sim_df = pd.DataFrame(user_sim, index=pivot.index, columns=pivot.index)
    similar_users = user_sim_df[user_input].sort_values(ascending=False)[1:6].index

    st.write(f"**Top 5 similar users to {user_input}:**")
    st.write(list(similar_users))
    # Recommend products rated by similar users but not by current user
    user_rated = set(pivot.loc[user_input][pivot.loc[user_input] > 0].index)
    recs = {}
    for sim_user in similar_users:
        products = pivot.loc[sim_user][pivot.loc[sim_user] > 0].index
        for prod in products:
            if prod not in user_rated:
                recs[prod] = recs.get(prod, 0) + pivot.loc[sim_user, prod]
    if recs:
        recs_sorted = sorted(recs.items(), key=lambda x: x[1], reverse=True)[:5]
        st.success("**Recommended Products (ProductId, Score):**")
        st.write(recs_sorted)
    else:
        st.warning("No recommendations found for this user. Try another user.")

# --- Interactive Chart ---
st.header("8. Interactive Visualization")
chart_type = st.selectbox("Choose Chart Type", ["Ratings per Product", "Average Ratings per Product"])
top_n = st.slider("Top N Products", 5, 30, 10)
if chart_type == "Ratings per Product":
    counts = top_n_counts(df, 'productId', n=top_n)
    st.bar_chart(counts)
else:
    avg = df.groupby('productId')['rating'].mean().sort_values(ascending=False).head(top_n)
    st.bar_chart(avg)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Python")