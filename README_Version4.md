# Product Recommendation System

This project demonstrates:
- Data sourcing from Kaggle
- Data cleaning and preprocessing
- Feature engineering and integrity checks
- Exploratory Data Analysis (EDA) and visualization
- Outlier handling and anomaly detection
- A collaborative-filtering recommendation system
- An interactive Streamlit dashboard

---

## 1. Dataset

Download the [Amazon Electronics Ratings dataset](https://www.kaggle.com/datasets/saurav9786/amazon-full-electronics-dataset) from Kaggle and place `ratings_Electronics.csv` in the project root.

---

## 2. Setup

Install requirements:
```bash
pip install -r requirements.txt
```

---

## 3. Run the Interactive Dashboard

```bash
streamlit run dashboard.py
```

---

## 4. Project Structure

- `dashboard.py` – Main Streamlit dashboard (EDA, visualization, recommendation)
- `eda_utils.py` – Utility functions for EDA and data cleaning
- `requirements.txt` – Required Python packages
- `README.md` – This documentation

---

## 5. Features

- **Data upload and summary**
- **Missing value and duplicate handling**
- **Summary statistics and visualizations:**
  - Ratings distribution
  - Top products/users
  - Anomaly and outlier detection
- **Collaborative filtering (user-based) recommendations**
- **Interactive charts and product recommendations**

---

## 6. References

- [Kaggle Dataset](https://www.kaggle.com/datasets/saurav9786/amazon-full-electronics-dataset)
- [Aryana12rj's GitHub Example](https://github.com/Aryana12rj/Product-Recommendation-System/blob/main/Recommendation%20System.ipynb)

---