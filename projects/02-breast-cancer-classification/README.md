# Breast Cancer Classification

A comparative binary classification study for distinguishing malignant and benign breast tumors using Logistic Regression, discriminant analysis, and Support Vector Machines.

---

## 📌 Overview

This project compares several machine learning classification methods on the **Breast Cancer Wisconsin (Diagnostic)** dataset.

The workflow includes exploratory data analysis, preprocessing, feature scaling, dimensionality reduction, hyperparameter tuning, model evaluation, confusion-matrix analysis, and ROC-AUC comparison.

---

## 📊 Dataset

The dataset contains:

- **569 samples**
- **30 numerical features**
- **212 malignant cases**
- **357 benign cases**

Target labels:

- `0` — Malignant
- `1` — Benign

The features describe characteristics of cell nuclei such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and related measurements.

---

## ⚙️ Workflow

1. Exploratory Data Analysis
2. Class-distribution and correlation analysis
3. Train–test split with stratification
4. Outlier handling using IQR-based capping
5. Feature standardization using `StandardScaler`
6. Logistic Regression with cross-validation
7. Linear Discriminant Analysis
8. QDA with PCA for dimensionality reduction
9. SVM with Linear, Polynomial, and RBF kernels
10. Hyperparameter tuning using GridSearchCV
11. Model comparison using multiple classification metrics
12. ROC and AUC analysis

---

## 🤖 Models

The following models are evaluated:

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis with PCA (QDA + PCA)
- Linear SVM
- Polynomial SVM
- RBF SVM

---

## 📐 Evaluation Metrics

Models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC

For Precision, Recall, and F1-score, the **malignant class is treated as the positive class**.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.9649** | 0.9750 | **0.9286** | **0.9512** | 0.9970 |
| Linear SVM | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9980 |
| RBF SVM | 0.9561 | 0.9744 | 0.9048 | 0.9383 | **0.9987** |
| QDA + PCA | 0.9474 | 0.9286 | **0.9286** | 0.9286 | 0.9888 |
| Polynomial SVM | 0.9474 | 0.9500 | 0.9048 | 0.9268 | 0.9937 |
| LDA | 0.9386 | **1.0000** | 0.8333 | 0.9091 | 0.9940 |

---

## 💡 Key Findings

- **Logistic Regression achieved the best overall performance**, with an accuracy of **96.49%** and malignant-class F1-score of **0.9512**.
- **RBF SVM achieved the highest AUC (0.9987)**.
- Logistic Regression and QDA + PCA achieved the highest malignant recall of **0.9286**.
- Linear SVM and RBF SVM produced identical test classification metrics.
- Polynomial SVM did not outperform the simpler Linear SVM.
- PCA reduced the original 30 features to **10 principal components** while retaining approximately **95.38% of the variance**, improving QDA stability.
- The results indicate that relatively simple linear models already provide strong separation for this dataset.

---

## 🧰 Technologies

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📁 File

- [`ArminAmani_Project2.ipynb`](./ArminAmani_Project2.ipynb) — Complete analysis, preprocessing, model training, hyperparameter tuning, and evaluation

The dataset is loaded directly through Scikit-learn, so no external dataset file is required.

---

> This project is intended as a machine learning classification study and is not designed for clinical diagnosis or medical decision-making.
