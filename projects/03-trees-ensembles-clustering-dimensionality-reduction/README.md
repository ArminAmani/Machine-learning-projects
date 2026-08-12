# Trees, Ensembles, Clustering & Dimensionality Reduction

A comprehensive machine learning study covering tree-based models, ensemble learning, clustering algorithms, and dimensionality-reduction techniques across multiple benchmark datasets.

---

## 📌 Overview

This project explores several major families of machine learning methods through classification, regression, clustering, and dimensionality-reduction tasks.

The analysis includes model tuning, cross-validation, feature-importance analysis, ensemble comparison, cluster-quality evaluation, and low-dimensional data visualization.

The project is organized into four main parts:

1. Decision Trees
2. Ensemble Methods
3. Clustering
4. Dimensionality Reduction

---

## 📊 Datasets

Three Scikit-learn datasets are used throughout the analysis:

| Dataset | Samples | Features | Main Task |
|---|---:|---:|---|
| Digits | 1,797 | 64 | Classification & Dimensionality Reduction |
| Diabetes | 442 | 10 | Regression |
| Iris | 150 | 4 | Clustering |

### Digits
Handwritten digit images represented as `8 × 8` pixel grids and flattened into 64 numerical features.

### Diabetes
Physiological measurements used to model a continuous disease-progression target.

### Iris
Four flower measurements representing three species: Setosa, Versicolor, and Virginica.

---

# 🌳 Part 1 — Decision Trees

Decision Tree models are applied to both classification and regression problems.

## Digits Classification

A `DecisionTreeClassifier` is tuned using cross-validation over tree depth, splitting criteria, and minimum samples per split.

### Best Configuration

- Criterion: `entropy`
- Maximum depth: `8`
- Minimum samples split: `5`
- Best CV Accuracy: `0.8539`

### Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.8500 |
| Macro Precision | 0.8544 |
| Macro Recall | 0.8498 |
| Macro F1-score | 0.8508 |

Feature-importance analysis is also used to identify the pixel regions most relevant to handwritten digit recognition.

---

## Diabetes Regression

A `DecisionTreeRegressor` is tuned using cross-validation.

### Best Configuration

- Criterion: `squared_error`
- Maximum depth: `2`
- Minimum samples split: `2`

### Test Performance

| Metric | Score |
|---|---:|
| MSE | 3735.50 |
| RMSE | 61.12 |
| R² | 0.2949 |

The feature-importance analysis identifies **BMI** as the dominant predictor, followed by **s5**.

The train/test R² analysis also demonstrates the bias-variance trade-off: deeper trees improve training performance while eventually reducing generalization performance.

---

# 🌲 Part 2 — Ensemble Methods

Several ensemble techniques are compared with the single Decision Tree.

The implemented methods include:

- Bagging
- Random Forest
- AdaBoost
- Gradient Boosting

Hyperparameters are optimized using cross-validation.

---

## Digits Classification Results

| Model | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Single Decision Tree | 0.8500 | 0.8508 |
| Bagging Classifier | 0.9306 | 0.9290 |
| AdaBoost | 0.8639 | 0.8652 |
| Gradient Boosting | 0.9611 | 0.9604 |
| **Random Forest** | **0.9639** | **0.9634** |

**Random Forest achieved the strongest overall classification performance.**

Its best configuration used:

- `n_estimators = 100`
- `max_depth = 11`
- `max_features = sqrt`

Random Forest feature importance was also mapped back to the original `8 × 8` image structure to identify informative pixel locations.

---

## Diabetes Regression Results

| Model | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Single Decision Tree | 3735.50 | 61.12 | 0.2949 |
| Gradient Boosting | 2774.59 | 52.67 | 0.4763 |
| **Random Forest** | **2728.02** | **52.23** | **0.4851** |

Both ensemble regressors significantly improved upon the single Decision Tree.

Random Forest again produced the strongest test performance among the evaluated tree-based regression models.

---

# 🔍 Part 3 — Clustering

Three unsupervised clustering approaches are compared on the standardized Iris dataset:

- K-Means
- Hierarchical Agglomerative Clustering
- Gaussian Mixture Model (GMM)

True species labels are used only for external evaluation and are not used to train the clustering algorithms.

The cluster structures are also visualized in two-dimensional PCA space.

---

## Clustering Results

| Algorithm | Silhouette | Davies-Bouldin | ARI | NMI |
|---|---:|---:|---:|---:|
| K-Means | 0.4599 | 0.8336 | **0.6201** | 0.6595 |
| Hierarchical Clustering | 0.4467 | **0.8035** | 0.6153 | **0.6755** |
| Gaussian Mixture Model | **0.4751** | 0.8867 | 0.5165 | 0.6571 |

### Key Observations

- **K-Means achieved the highest Adjusted Rand Index (ARI).**
- **Hierarchical Clustering achieved the highest NMI and lowest Davies-Bouldin Index.**
- **GMM achieved the highest Silhouette Score.**
- Setosa forms a relatively distinct group, while Versicolor and Virginica show greater overlap.

The analysis demonstrates that different clustering metrics can favor different algorithms depending on the geometric and statistical properties being evaluated.

---

# 📉 Part 4 — Dimensionality Reduction

PCA and t-SNE are applied to the standardized Digits dataset to investigate how well high-dimensional image information can be represented in two dimensions.

---

## PCA

The Digits dataset contains 64 original pixel features.

PCA analysis shows that:

- **31 principal components** are required to retain approximately **90% of the total variance**
- The first two principal components retain only approximately **21.6% of the variance**

The 2D PCA representation therefore shows considerable overlap among digit classes.

---

## t-SNE

Two nonlinear t-SNE representations are generated using:

- Perplexity = `30`
- Perplexity = `100`

Compared with PCA, both t-SNE embeddings provide much clearer local separation between digit classes.

---

## Quantitative 2D Representation Comparison

A K-Nearest Neighbors classifier (`k = 5`) is used to compare the class information preserved in each two-dimensional representation.

| Representation | KNN Accuracy |
|---|---:|
| PCA 2D | 0.5278 |
| t-SNE 2D — Perplexity 100 | 0.9472 |
| **t-SNE 2D — Perplexity 30** | **0.9667** |

The results show substantially stronger class separation in the t-SNE embeddings than in the two-dimensional PCA representation.

---

## 💡 Key Findings

- Decision Trees provide interpretable models but are highly sensitive to tree complexity.
- Ensemble methods substantially improve the predictive performance and stability of individual trees.
- **Random Forest produced the strongest overall performance** for both the Digits classification and Diabetes regression tasks.
- Gradient Boosting achieved classification performance very close to Random Forest.
- Different clustering algorithms perform best according to different internal and external evaluation metrics.
- K-Means and Hierarchical Clustering showed stronger agreement with the true Iris species than GMM in terms of ARI/NMI.
- PCA provides a fast and interpretable linear representation of variance.
- t-SNE provides much clearer nonlinear visualization of local class structure in the Digits dataset.
- Model performance and representation quality depend strongly on the assumptions and objectives of each algorithm.

---

## 🧰 Technologies

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

### Machine Learning Techniques

- Decision Trees
- Bagging
- Random Forest
- AdaBoost
- Gradient Boosting
- GridSearchCV
- Cross-Validation
- K-Means Clustering
- Hierarchical Clustering
- Gaussian Mixture Models
- Principal Component Analysis (PCA)
- t-SNE
- K-Nearest Neighbors

---

## 📁 File

- [`ArminAmani_Project3.ipynb`](./ArminAmani_Project3.ipynb) — Complete implementation, model tuning, evaluation, visualization, and comparative analysis

All datasets are loaded directly through Scikit-learn, so no external dataset files are required.
