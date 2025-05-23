import pandas as pd
import numpy as np

def clean_data(df):
    # Drop timestamp if present
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    # Ensure correct dtypes
    df['rating'] = df['rating'].astype(float)
    return df

def get_summary_stats(df):
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_unique_users": df['userId'].nunique(),
        "n_unique_products": df['productId'].nunique(),
        "rating_stats": df['rating'].describe(),
    }
    return summary

def get_missing_values(df):
    return df.isnull().sum()

def top_n_counts(df, col, n=10):
    return df[col].value_counts().head(n)

def add_features(df):
    df['rating_count'] = df.groupby('productId')['rating'].transform('count')
    df['avg_rating'] = df.groupby('productId')['rating'].transform('mean')
    return df

def get_outliers(df):
    q1 = df['rating'].quantile(0.25)
    q3 = df['rating'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    outliers = df[(df['rating'] < lower) | (df['rating'] > upper)]
    return outliers

def filter_for_demo(df, max_rows=20000):
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df