# Classical Machine Learning

## What is Classical Machine Learning?

**Classical Machine Learning (ML)** refers to a set of algorithms and techniques
that allow computers to **learn patterns from data** and make decisions or predictions without being explicitly programmed for every rule.
It focuses on **statistical methods and mathematical models** to learn relationships in data.

Unlike Deep Learning, Classical ML works well with **smaller datasets**, is often
more **interpretable**, and requires **feature engineering**.

---

## Classical ML vs Deep Learning

| **Feature**         | **Classical ML**                       | **Deep Learning**            |
| ------------------- | -------------------------------------- | ---------------------------- |
| Data Requirement    | Small to medium datasets               | Large datasets required      |
| Feature Engineering | Manual                                 | Automatic (learned features) |
| Interpretability    | High                                   | Low (black-box models)       |
| Training Time       | Faster                                 | Slower                       |
| Hardware Needs      | Low (CPU sufficient)                   | High (GPU/TPU preferred)     |
| Examples            | Linear Regression, SVM, Decision Trees | CNNs, RNNs, Transformers     |

- **Classical ML** is like a skilled analyst who uses **rules and logic** to make decisions.
- **Deep Learning** is like a brain that learns patterns **automatically from experience**.

---

## Types of Machine Learning

### Supervised Learning

- Learns from **labeled data**
- Predicts output based on input-output pairs

**Examples:**

- Classification (Spam detection)
- Regression (House price prediction)

---

### Unsupervised Learning

- Works with **unlabeled data**
- Finds hidden patterns or structures

**Examples:**

- Clustering (Customer segmentation)
- Dimensionality Reduction (PCA)

---

### Semi-Supervised Learning

- Mix of **labeled + unlabeled data**
- Useful when labeling data is expensive

---

### Reinforcement Learning (Classical Perspective)

- Learns via **rewards and penalties**
- Used in control systems and decision-making

---

## Common Algorithms

### Regression

- Linear Regression
- Ridge / Lasso Regression

---

### Classification

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Naive Bayes

---

### Tree-Based Models

- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)

---

### Clustering

- K-Means
- Hierarchical Clustering
- DBSCAN

---

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- t-SNE

---

## Workflow of Classical ML

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Selection
5. Training
6. Evaluation
7. Deployment

<!-- ---

## Popular Tools & Libraries

- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}
- :contentReference[oaicite:2]{index=2}
- :contentReference[oaicite:3]{index=3}
- :contentReference[oaicite:4]{index=4} -->

---

## Use of Classical ML

<!-- tabs:start -->

#### **Use Classical ML When**

- Dataset is **small or medium-sized**
- Need **interpretability**
- Running on **edge/embedded systems**
- Limited compute resources
- Problem is well-structured

#### **Avoid Classical ML When**

- Data is **unstructured** (images, audio, text at scale)
- Need **high-level abstraction learning**
- Working with **very large datasets**

<!-- tabs:end -->

---

## Common Use Cases

- Fraud Detection
- Recommendation Systems
- Predictive Maintenance
- Medical Diagnosis
- Customer Segmentation
- Time Series Forecasting

---

## Limitations

- Requires **manual feature engineering**
- Performance depends heavily on **data quality**
- Struggles with **high-dimensional unstructured data**
