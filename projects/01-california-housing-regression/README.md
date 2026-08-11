# California Housing Value Prediction

A comparative regression study focused on predicting median housing values using linear, regularized, and polynomial regression models on the California Housing dataset.

---

## 📌 Overview

This project explores the performance of several regression techniques for modeling housing values based on demographic, geographic, and housing-related features.

The workflow includes exploratory data analysis, preprocessing, regularization, cross-validation, coefficient analysis, and nonlinear feature expansion.

The main objective is to compare how different regression formulations behave in terms of predictive performance, model complexity, and coefficient stability.

---

## 📊 Dataset

The analysis uses the **California Housing dataset**, containing:

- **20,640 observations**
- **8 numerical input features**
- **Target:** Median housing value

### Features

- `MedInc` — Median income
- `HouseAge` — Median house age
- `AveRooms` — Average rooms per household
- `AveBedrms` — Average bedrooms per household
- `Population` — Block-group population
- `AveOccup` — Average household occupancy
- `Latitude`
- `Longitude`

---

## 🔍 Workflow

The project follows this general workflow:

1. Exploratory Data Analysis
2. Distribution and correlation analysis
3. Outlier inspection and preprocessing
4. Feature transformation and standardization
5. Train–test split
6. Baseline linear regression
7. Ridge regression with cross-validation
8. Lasso regression with cross-validation
9. Coefficient comparison
10. Residual analysis
11. Polynomial feature expansion
12. Final model comparison

---

## 🤖 Models

The following regression models are evaluated:

- Ordinary Least Squares Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Linear Regression
- Polynomial Ridge Regression
- Polynomial Lasso Regression

---

## 📐 Evaluation Metrics

Model performance is evaluated using:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination (R²)**

---

## 📈 Results

| Model | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|
| OLS Linear Regression | 0.468183 | 0.684239 | 0.653678 |
| Ridge Regression | 0.468181 | 0.684237 | 0.653681 |
| Lasso Regression | 0.468176 | 0.684234 | 0.653684 |
| Polynomial Linear Regression | 0.388131 | 0.623002 | 0.712894 |
| **Polynomial Ridge Regression** | **0.388058** | **0.622943** | **0.712948** |
| Polynomial Lasso Regression | 0.390580 | 0.624964 | 0.711083 |

---

## 💡 Key Findings

- Standard linear, Ridge, and Lasso models produced very similar predictive performance.
- Regularization had only a minor effect on predictive accuracy in the original feature space.
- Polynomial feature expansion resulted in a noticeable improvement in model performance.
- Polynomial Ridge Regression achieved the lowest test MSE and highest test R² among the evaluated models.
- Geographic variables and median income showed strong influence within the fitted linear models.
- The results suggest that nonlinear interactions between features provide useful predictive information beyond a purely linear formulation.

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

## 📁 Files

- [`ArminAmani_Project1.ipynb`](./ArminAmani_Project1.ipynb) — Complete analysis and model implementation
- [`california_housing.xlsx`](./california_housing.xlsx) — Dataset used in the analysis

---

## 🎯 Project Scope

This project demonstrates a complete applied regression workflow, including:

- exploratory analysis
- data preprocessing
- regularization
- cross-validation
- model comparison
- coefficient analysis
- nonlinear feature engineering
- performance evaluation
