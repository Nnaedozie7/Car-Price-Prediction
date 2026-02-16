# Car Price Prediction

A machine learning regression project focused on predicting car prices using structured vehicle specifications. The notebook demonstrates an end-to-end workflow including data loading, cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## 📌 Project Overview

The goal of this project is to build a regression model that estimates a car’s **price** from a mix of numerical and categorical attributes (e.g., brand, engine characteristics, fuel system, body type).  

This notebook covers:

- Data cleaning and preprocessing  
- Brand extraction and normalization from the car name  
- Exploratory visualizations and basic statistical checks  
- Feature engineering (including brand-level price aggregation)  
- Regression modeling using **Linear Regression**  
- Evaluation using error metrics and **R²**

---

## 📊 Dataset

The notebook loads the dataset directly from an online source:

- `CarPrice_Assignment.csv` (loaded via a URL inside the notebook)

The dataset contains vehicle-level attributes such as:

- Car brand (extracted from `CarName`)
- Engine and fuel characteristics
- Body style / car type attributes
- Numerical technical specifications
- **Price** (target variable)

**Target:** `price`

---

## 🔎 Exploratory Data Analysis (EDA)

The EDA process includes:

- Reviewing data types, summary statistics, missing values, and duplicates  
- Exploring car brand distributions (count plot)  
- Inspecting key feature behavior and relationships with the target variable  
- Checking for distribution / normality behavior (where applicable)

EDA insights guide the feature preparation steps used later in modeling.

---

## ⚙️ Data Preprocessing & Feature Engineering

Key preprocessing and feature engineering steps implemented in the notebook:

- Extracting `brand` from `CarName` and normalizing inconsistent spellings (e.g., “vw” → “volkswagen”)  
- Dropping non-essential columns (e.g., identifiers and raw name fields)  
- Creating a **brand average price** feature by grouping and merging brand-level mean price  
- Creating categorical groupings derived from brand average pricing (brand category)  
- Handling categorical variables via:
  - Manual dummy encoding, and/or  
  - `OneHotEncoder` using `ColumnTransformer` inside a `Pipeline`
- Train/test split using `train_test_split`
- Feature scaling using `StandardScaler`

---

## 🤖 Model Development

The notebook trains and evaluates **Linear Regression** using:

- A standard approach with `StandardScaler` + `LinearRegression`
- A scikit-learn **Pipeline** for cleaner preprocessing + modeling
- A Pipeline variant using:
  - `ColumnTransformer` + `OneHotEncoder` for categorical features
  - `StandardScaler`
  - `LinearRegression`

✅ **Models used in the notebook:**  
- **Linear Regression** (including pipeline-based implementations)

---

## 📈 Model Evaluation

Model performance is evaluated using:

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

These metrics are computed on the test split to estimate generalization performance.

---

## 🛠️ Technologies & Tools

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- SciPy  
- Scikit-learn  
- Jupyter Notebook  

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
