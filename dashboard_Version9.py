import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Set up Streamlit page
st.set_page_config(page_title="Product Recommendation System", layout="wide")
sns.set_theme(style="whitegrid", palette="muted")

# ---- Sidebar: Dataset Info & Upload ----
st.sidebar.header("Step 1: Upload Dataset")
st.sidebar.markdown("""
Dataset: [Amazon Electronics Ratings (Kaggle)](https://www.kaggle.com/datasets/saurav9786/amazon-full-electronics-dataset)
""")
uploaded_file = st.sidebar.file_uploader("Upload ratings_Electronics.csv", type="csv")
if not uploaded_file:
    st.info("Please upload the ratings_Electronics.csv file to begin.")
    st.stop()

# ---- Data Loading & Cleaning ----
def clean_data(df):
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    df = df.drop_duplicates()
    df['rating'] = df['rating'].astype(float)
    return df

def add_features(df):
    df['rating_count'] = df.groupby('productId')['rating'].transform('count')
    df['avg_rating'] = df.groupby('productId')['rating'].transform('mean')
    return df

def get_outliers(df):
    q1 = df['rating'].quantile(0.25)
    q3 = df['rating'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df['rating'] < lower) | (df['rating'] > upper)]

def filter_for_demo(df, max_rows=20000):
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df

# Load and prepare data
df = pd.read_csv(uploaded_file, names=['userId', 'productId', 'rating', 'timestamp'])
df = clean_data(df)
df = add_features(df)

# ---- 1. Data Cleaning & Integrity ----
st.title("📦 Product Recommendation System Dashboard")
st.header("1. Data Cleaning & Integrity Checks")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Unique Users", df['userId'].nunique())
col3.metric("Unique Products", df['productId'].nunique())
st.write("**Missing values per column:**")
st.write(df.isnull().sum())
st.write("**Duplicate rows removed:**", df.duplicated().sum())
st.markdown("""
#### **Interpretation**
- No missing values and duplicate records are present. Ratings are on a 1-5 scale. Data types are consistent.
""")

# ---- 2. Feature Engineering & Insights ----
st.header("2. Feature Engineering & Insights")
st.write(df.head())
st.markdown("""
**Interpretation:**  
- `rating_count` shows number of ratings per product.  
- `avg_rating` is the average rating for each product.
""")

# ---- 3. Summary Statistics & Patterns ----
st.header("3. Summary Statistics & Patterns")
stats = df['rating'].describe()
st.write(stats)
st.markdown(f"""
- **Most common rating:** {df['rating'].mode()[0]}  
- **Median rating:** {stats['50%']}  
- **Ratings skewed:** {'Yes' if stats['mean'] > stats['50%'] else 'No'}
""")

# ---- 4. Data Visualizations & Trends ----
st.header("4. Data Visualizations & Trends")
viz_choice = st.selectbox(
    "Choose a chart type for insight:",
    [
        "Ratings Distribution (Histogram)",
        "Boxplot of Ratings",
        "Top N Rated Products (Bar Chart)",
        "Top N Most Active Users (Bar Chart)",
        "User-Product Ratings Heatmap (Sample)"
    ]
)

if viz_choice == "Ratings Distribution (Histogram)":
    st.markdown("**Histogram:** Most ratings are 4 or 5, indicating user positivity.")
    fig, ax = plt.subplots()
    sns.histplot(df['rating'], bins=5, kde=True, ax=ax)
    ax.set_xlabel("Rating")
    st.pyplot(fig)
    st.markdown("**Interpretation:** Ratings are right-skewed; most users rate products highly.")

elif viz_choice == "Boxplot of Ratings":
    st.markdown("**Boxplot:** Visualizing outliers and spread.")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['rating'], ax=ax)
    st.pyplot(fig)
    st.markdown("**Interpretation:** The boxplot confirms a positive skew with few outliers.")

elif viz_choice == "Top N Rated Products (Bar Chart)":
    top_n = st.slider("Top N products", 5, 30, 10)
    prod_counts = df['productId'].value_counts().head(top_n)
    fig, ax = plt.subplots()
    sns.barplot(x=prod_counts.values, y=prod_counts.index, ax=ax, orient='h')
    ax.set_xlabel("Ratings Count")
    ax.set_ylabel("Product ID")
    st.pyplot(fig)
    st.markdown("**Interpretation:** These products are the most popular (by rating frequency).")

elif viz_choice == "Top N Most Active Users (Bar Chart)":
    top_n = st.slider("Top N users", 5, 30, 10)
    user_counts = df['userId'].value_counts().head(top_n)
    fig, ax = plt.subplots()
    sns.barplot(x=user_counts.values, y=user_counts.index, ax=ax, orient='h')
    ax.set_xlabel("Ratings Count")
    ax.set_ylabel("User ID")
    st.pyplot(fig)
    st.markdown("**Interpretation:** Power users who contribute most ratings.")

elif viz_choice == "User-Product Ratings Heatmap (Sample)":
    st.markdown("**Heatmap:** Sample of user-product ratings (blue = more ratings).")
    small = df.pivot_table(index="userId", columns="productId", values="rating").sample(20, axis=0).sample(20, axis=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(small, cmap="YlGnBu", cbar=True, ax=ax)
    st.pyplot(fig)
    st.markdown("**Interpretation:** Sparse matrix is typical in recommender systems.")

# ---- 5. Outlier & Anomaly Detection ----
st.header("5. Outlier & Anomaly Detection")
outliers = get_outliers(df)
st.write(f"**Number of rating outliers detected:** {len(outliers)}")
if not outliers.empty:
    st.dataframe(outliers.sample(min(10, len(outliers))))
st.markdown("**Interpretation:** Outliers are rare, confirming the reliability of most ratings.")

# ---- 6. Interactive Data Explorer ----
st.header("6. Interactive Data Explorer")
st.dataframe(df.sample(50), use_container_width=True)

# ---- 7. Product Recommendation Engine ----
st.header("7. Product Recommendation Engine")
st.markdown("""
Select a user to get personalized recommendations based on similar users' ratings.
""")
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
        st.markdown("""
        **Interpretation:**  
        - These are products highly rated by users similar to the selected user, but not yet rated by them.
        """)
    else:
        st.warning("No recommendations found for this user. Try another user.")

# ---- 8. Data Storytelling: Key Insights ----
st.header("8. Data Storytelling: Key Insights")
st.markdown("""
- Most users give high ratings, indicating satisfaction bias.
- Only a few products and users dominate ratings activity — classic long-tail behavior.
- Outliers are rare, so recommendations are based on reliable data.
- The data's sparsity supports the use of collaborative filtering.
- Interactive visualizations and recommendation tools help you explore and act on these insights.
""")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Python | Aryana12rj/Product-Recommendation-System")