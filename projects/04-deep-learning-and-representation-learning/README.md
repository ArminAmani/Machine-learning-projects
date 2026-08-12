# Deep Learning & Representation Learning

A comprehensive deep learning study covering neural-network regularization, optimization, convolutional neural networks, recurrent neural networks, and neural word representations across numerical, image, text, and domain-specific data.

---

## 📌 Overview

This project explores several major concepts in deep learning through a series of practical experiments implemented with **TensorFlow/Keras**.

The analysis includes model regularization, optimizer comparison, image classification, sentiment analysis, representation learning, learning-curve analysis, confusion matrices, ROC-AUC evaluation, and embedding visualization.

The project is organized into four main parts:

1. Regularization for Neural Networks
2. Optimization for Deep Networks
3. CNN & LSTM Applications
4. CBOW Word Embeddings

---

## 📊 Datasets

Several datasets and data sources are used throughout the analysis:

| Dataset / Data           |           Size | Main Task                    |
| ------------------------ | -------------: | ---------------------------- |
| Synthetic Regression     |  1,000 samples | Nonlinear Regression         |
| MNIST                    |  70,000 images | Digit Classification         |
| CIFAR-10                 |  60,000 images | Image Classification         |
| IMDB                     | 50,000 reviews | Sentiment Classification     |
| Aerodynamics Text Corpus |    1,000 words | Word Representation Learning |

---

# 🧠 Part 1 — Regularization for Neural Networks

A multilayer perceptron is trained on a synthetic nonlinear regression problem with additive random noise.

## Problem Definition

The target function is defined as:

<p align="center">
  <strong>f(x<sub>1</sub>, x<sub>2</sub>) = x<sub>1</sub><sup>3</sup> − x<sub>1</sub><sup>2</sup> + 2x<sub>2</sub><sup>2</sup> + 3x<sub>1</sub>x<sub>2</sub> + 5</strong>
</p>

The processed dataset is divided into:

* 720 training samples
* 80 validation samples
* 200 test samples

The baseline MLP architecture is:

```text
2 → 10 → 10 → 1
```

with ReLU activation in both hidden layers, a linear output layer, MSE loss, and the Adam optimizer.

---

## Baseline MLP Performance

| Metric    |     Result |
| --------- | ---------: |
| Test MSE  |  24,199.43 |
| Test RMSE |     155.56 |
| Test MAE  |     122.70 |
| Test R²   | **0.8474** |

The model explains approximately **84.7% of the variance** in the test targets.

---

## Regularization Comparison

Two strategies are investigated:

* Early Stopping with `patience = 10`
* Dropout with `p = 0.5` after each hidden layer

| Strategy           | Epochs Completed | Best Validation Epoch |      Test MSE |
| ------------------ | ---------------: | --------------------: | ------------: |
| **Early Stopping** |              200 |                   200 | **24,199.43** |
| Dropout            |              200 |                   200 |     48,052.82 |

Early stopping was not activated because the validation loss continued improving through epoch 200.

The dropout model produced substantially higher test error, indicating that a dropout rate of `0.5` imposed excessive regularization on the relatively small network.

---

# ⚙️ Part 2 — Optimization on MNIST

Two optimization strategies are compared using identical MLP architectures and the same initial model weights:

* SGD with Momentum
* Adam

The MNIST data is split into:

* 54,000 training images
* 6,000 validation images
* 10,000 test images

The network architecture is:

```text
784 → 128 → 64 → 10
```

with ReLU hidden layers and a Softmax output layer.

---

## Optimizer Comparison

| Optimizer      | Test Accuracy |  Test Loss | Epoch Reaching 97% Validation Accuracy |
| -------------- | ------------: | ---------: | -------------------------------------: |
| SGD + Momentum |        97.61% | **0.0815** |                                     10 |
| **Adam**       |    **97.79%** |     0.1016 |                                  **4** |

Adam reached high validation accuracy considerably faster and achieved slightly higher test accuracy.

However, SGD with Momentum produced:

* Lower test loss
* Lower late-stage validation-loss variability
* More stable behavior near the end of training

The experiment illustrates that optimizer quality should not be evaluated only by final accuracy; **convergence speed, loss behavior, and stability** are also important.

---

# 🖼️ Part 3 — CNN & LSTM Applications

Two neural-network architectures are applied to different data modalities:

* CNN for image classification on CIFAR-10
* LSTM for sentiment classification on IMDB reviews

---

## CNN on CIFAR-10

The CIFAR-10 dataset contains RGB images from 10 classes.

The data is divided into:

* 45,000 training images
* 5,000 validation images
* 10,000 test images

The model uses:

* Two convolutional layers with 32 and 64 filters
* Max Pooling
* Dropout (`0.25`)
* Dense layer with 512 neurons
* Dropout (`0.50`)
* Softmax output layer
* Horizontal flipping and random translation for data augmentation

The network contains **6,447,562 parameters**.

### CNN Performance

| Metric         |     Result |
| -------------- | ---------: |
| Test Accuracy  | **78.14%** |
| Test Loss      |     0.6580 |
| Macro F1-score |     0.7776 |
| Macro ROC-AUC  | **0.9754** |
| Micro ROC-AUC  | **0.9766** |

The strongest class-level ROC-AUC was obtained for **automobile (`0.9933`)**, while **cat (`0.9390`)** was the most challenging according to this metric.

The notebook also includes:

* Training and validation learning curves
* Classification report
* Confusion matrix
* One-vs-rest ROC curves
* Class-level ROC-AUC analysis
* Visualization of misclassified images

---

## LSTM on IMDB Sentiment

An LSTM network is used for binary sentiment classification of movie reviews.

The processed data contains:

* 22,500 training reviews
* 2,500 validation reviews
* 25,000 test reviews
* Vocabulary size: `20,000`
* Sequence length: `200`

The architecture consists of:

```text
Embedding(128)
      ↓
LSTM(128)
      ↓
Dense(64, ReLU)
      ↓
Dropout(0.5)
      ↓
Dense(1, Sigmoid)
```

The model contains **2,699,905 parameters**.

### LSTM Performance

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy |     78.84% |
| Precision     |     73.63% |
| Recall        | **89.86%** |
| F1-score      | **80.94%** |
| Test Loss     |     1.6217 |

The final training accuracy reached **99.69%**, compared with **80.76% validation accuracy**.

The best validation performance occurred around epoch 2, followed by a growing train-validation gap, providing clear evidence of **overfitting**.

Error analysis also showed:

* False Positives: 4,023
* False Negatives: 1,267
* Misclassified Reviews: 5,290

The model therefore achieved high positive-class recall while producing a larger number of false-positive predictions.

---

# 🔤 Part 4 — CBOW Word Embeddings

Continuous Bag-of-Words (CBOW) models are used to learn word representations from a small **aerodynamics-focused text corpus**.

After preprocessing:

* Corpus size: 1,000 words
* Final vocabulary size: 141
* CBOW training examples: 996
* Context: two words before and two words after the target

The CBOW structure follows:

```text
[w(t−2), w(t−1), w(t+1), w(t+2)] → target word
```

Two representations are investigated:

* Direct 2D embeddings
* 10D embeddings followed by PCA projection to 2D

---

## CBOW Results

| Model          | Parameters | Final Loss | Final Accuracy |
| -------------- | ---------: | ---------: | -------------: |
| Direct 2D CBOW |        705 |     3.1793 |         31.33% |
| **10D CBOW**   |      2,961 | **2.0066** |     **48.90%** |

The recorded results show stronger target-word prediction performance for the 10D representation.

---

## PCA Projection

The learned 10D embeddings are projected into two dimensions using PCA.

| Component | Explained Variance |
| --------- | -----------------: |
| PC1       |             16.36% |
| PC2       |             14.95% |
| Total     |         **31.31%** |

The mean top-5 nearest-neighbor overlap between the original 10D embeddings and their PCA projection is approximately **11.49%**.

This demonstrates that PCA provides a useful visualization of the higher-dimensional representation, but a substantial amount of local embedding structure is lost during projection.

---

## 💡 Key Findings

* The baseline MLP achieved strong predictive performance on the nonlinear regression problem with a test R² of **0.8474**.
* Early stopping preserved performance but was not activated because validation loss continued improving through the final epoch.
* Strong dropout degraded regression performance, demonstrating the effect of excessive regularization on a small neural network.
* Adam converged substantially faster on MNIST and achieved slightly higher test accuracy than SGD with Momentum.
* SGD with Momentum produced lower test loss and more stable late-stage validation behavior.
* The CIFAR-10 CNN achieved **78.14% test accuracy** and strong class discrimination with a **0.9754 macro ROC-AUC**.
* The IMDB LSTM achieved high positive-class recall but showed substantial overfitting during extended training.
* The higher-dimensional CBOW representation achieved better recorded target-word prediction performance than the direct 2D representation.
* PCA provides an interpretable visualization of word embeddings but does not fully preserve the geometric structure of the original higher-dimensional space.
* Learning curves, error analysis, ROC-AUC, confusion matrices, and representation analysis provide important information beyond a single performance metric.

---

## 🧰 Technologies

* Python
* Jupyter Notebook
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

### Deep Learning & Analysis Techniques

* Multilayer Perceptrons
* Early Stopping
* Dropout
* SGD with Momentum
* Adam Optimization
* Convolutional Neural Networks
* Data Augmentation
* LSTM Networks
* Word Embeddings
* Continuous Bag-of-Words (CBOW)
* Principal Component Analysis (PCA)
* Confusion-Matrix Analysis
* ROC-AUC Evaluation

---

## 📁 File

* [`ArminAmani_Project4.ipynb`](./ArminAmani_Project4.ipynb) — Complete implementation, neural-network training, evaluation, visualization, and comparative analysis

The datasets and text corpus used in the experiments are acquired programmatically by the notebook, so no separate dataset files are included in this project directory.
