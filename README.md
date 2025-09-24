# ❤️ Heart Disease Prediction – Machine Learning Full Pipeline

A **comprehensive end-to-end machine learning project** built on the [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease).  
This project takes you through **data cleaning, feature selection, dimensionality reduction, supervised & unsupervised learning, model optimization, and deployment** with a **Streamlit web UI**.

---

## 📋 Project Overview

This project aims to **analyze, predict, and visualize heart disease risk** using modern machine learning techniques.  
It demonstrates a **complete workflow** from data preprocessing to model deployment, making it a perfect reference for students, researchers, and ML practitioners.

### 🔑 Objectives

- ✅ Perform data preprocessing & cleaning (handle missing values, encoding, scaling)
- ✅ Apply PCA for dimensionality reduction
- ✅ Implement feature selection (RFE, Chi-Square, Feature Importance)
- ✅ Train classification models (Logistic Regression, Decision Tree, Random Forest, SVM)
- ✅ Apply clustering techniques (K-Means, Hierarchical Clustering)
- ✅ Perform hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- ✅ Build a Streamlit web UI for real-time predictions *(Bonus)*
- ✅ Deploy with Ngrok for public access *(Bonus)*

---

## 🛠️ Tech Stack

**Languages:**  
- Python 🐍  

---

## 📂 Project Structure
``` bash
Heart_Disease_Project/
│── data/
│ ├── heart_disease.csv
| ├── cleaned_heart_disease.csv
| ├── heart_disease_PCA.csv
| ├── heart_disease_feature_selection.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│── models/
│ └── final_model.pkl
│── ui/
│ └── app.py # Streamlit UI
│── deployment/
│ └── ngrok_setup.txt
│── results/
│ └── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/abdoashraf952/Heart_Disease_Project.git
cd Heart_Disease_Project
```
### 2️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebooks 

Explore step-by-step notebooks inside `/notebooks`.

### 4️⃣ Run the Streamlit App

``` bash
streamlit run ui/app.py
```

------------------------------------------------------------------------

## 📊 Results

-   ✅ **Cleaned & preprocessed dataset**
-   ✅ **PCA-transformed data with variance visualization**
-   ✅ **Trained supervised models with evaluation metrics**
-   ✅ **Clustering analysis with visualization**
-   ✅ **Best-performing model with hyperparameter tuning**
-   ✅ **Deployed web app for real-time predictions**

