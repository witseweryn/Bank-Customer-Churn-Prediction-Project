# 🏦 Bank Customer Churn Prediction

> Predicting customer churn using machine learning models to support business decisions in customer retention strategies.

---

## 📌 Project Overview

Customer retention is a key performance driver in the banking sector. This project focuses on **predictive modeling of customer churn** using structured bank data and machine learning techniques.

The goal is not only to predict whether a customer will churn, but also to **understand which factors drive churn**, and how those insights can support **data-informed business strategies**.

---

## 📁 Dataset

- **Target Variable:** `churn` (1 = left the bank, 0 = stayed)
- **Features:** 11 customer attributes including demographics, account balance, engagement, and activity level

### Key Attributes:
- `credit_score`, `age`, `balance`, `estimated_salary`
- `country`, `gender`, `active_member`, `products_number`
- **No missing values** or duplicates.

---

## 🧭 Business Context

Customer churn directly impacts revenue. Acquiring new customers is significantly more expensive than retaining existing ones. Therefore, **understanding why customers churn** is critical to:

- **Increase customer lifetime value (CLV)**
- **Improve retention strategies**
- **Allocate marketing efforts more effectively**

This project answers:  
❓ *Which customers are at the highest risk of churning?*  
❓ *What are the most influential factors driving churn?*  
❓ *How accurate are predictive models in identifying these customers?*  

---

## 🔍 Exploratory Data Analysis (EDA)

- **Imbalanced dataset**: 20% churn rate
- **Churn more frequent among:**
  - Older customers
  - Inactive members
  - German customers

### Key Insights:

| Finding | So what? |
|--------|----------|
| German customers churn significantly more | Localized marketing or product tailoring may be needed |
| Inactive customers churn more | Engagement strategies (e.g. loyalty programs, proactive support) could reduce churn |
| Older customers more likely to churn | Possibly due to lower tech affinity or changing life stages—require different retention messaging |

---

## 📊 Visualizations

- Churn Distribution
- Age vs Churn (Boxplot)
- Country-wise Churn Percentages
- Correlation Heatmap
- Feature Importances (Random Forest)
- ROC Curve

All visualizations aim to combine **statistical validity with business interpretability**.

---

## ⚙️ Preprocessing

- Dropped `customer_id` (non-informative)
- One-hot encoded categorical features
- Feature scaling using `StandardScaler`
- Train-test split: 80/20

---

## 🤖 Models & Evaluation

| Model | Accuracy | Recall (Churn) | F1-Score (Churn) |
|-------|----------|----------------|------------------|
| Logistic Regression | 81.1% | 20% | 0.29 |
| Random Forest | 86.6% | 47% | 0.58 |
| **Optimized RF (GridSearchCV)** | **87.1%** | **48%** | **0.59** |

### Why Random Forest?

- Handles imbalanced data better
- Captures non-linear relationships
- Provides **feature importance** for business insight

### ROC-AUC:  
Achieved **AUC ≈ 0.87**, indicating strong class separation ability.

---

## 🔬 Feature Importance (Top 5)

| Feature | Importance |
|---------|------------|
| `age` | Most influential |
| `estimated_salary` | High predictor |
| `credit_score` | Strong indicator |
| `balance` | Related to engagement |
| `products_number` | Key behavioral signal |

**So what?**  
These are **actionable insights**. The bank can target these areas with:

- Retention campaigns for high-risk age groups
- Better credit offerings for at-risk scores
- Personalized outreach based on product holding patterns

---

## 🧪 Hyperparameter Tuning

Used `GridSearchCV` to fine-tune the Random Forest with:

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

---

## ✅ Key Takeaways

- ✅ **Random Forest outperformed Logistic Regression**, especially for minority class recall
- 📊 **Age, activity level, and geography** are strong churn drivers
- 🔍 **GridSearchCV** improved performance modestly
- 💡 **Business value lies not just in prediction**, but in using these insights to shape strategy

---

## 📈 Business Application

This model can be integrated into:

- 🧠 **CRM systems** for real-time churn risk scoring  
- 🎯 **Customer segmentation** for proactive engagement  
- 💰 **Targeted retention campaigns** with higher ROI  

---

## 🚀 Next Steps

- ⚖️ Apply **SMOTE** or **class weighting** to address class imbalance  
- ⚡ Explore **XGBoost** or **LightGBM** for better model performance  
- 🖥️ Deploy as a **REST API** or **Streamlit dashboard** for business use  
- 🔄 Implement **continuous monitoring** with real-world feedback loop  

---

## 🛠️ Technologies Used

- 🐍 **Python** (`pandas`, `numpy`, `sklearn`, `seaborn`, `matplotlib`)  
- 📦 **Scikit-learn** (modeling, evaluation, hyperparameter tuning)  
- 📓 **Jupyter Notebooks** for interactive EDA and experimentation  


Data Source: https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset
