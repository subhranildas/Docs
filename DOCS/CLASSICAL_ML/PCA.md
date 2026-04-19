# Principal Component Analysis (PCA)

## What is PCA?

**Principal Component Analysis (PCA)** is an **unsupervised learning algorithm**
used for **dimensionality reduction**.

> It transforms data into a new coordinate system where:
>
> - The **most important information (variance)** is captured first
> - Redundant information is removed

---

## Why Do We Need PCA?

Real-world datasets often have:

- Too many features
- Redundant or correlated features
- Noise

---

### Problems Without PCA

- Slow computation
- Overfitting
- Difficult visualization
- Poor model performance

---

### PCA Solves This By

- Reducing dimensions
- Removing redundancy
- Keeping maximum information

---

## Real-Life Analogy

Imagine taking a photo of a 3D object:

- Different angles show different information
- Best angle captures most details

> PCA finds the **best angle (direction)** to view the data

---

## Key Idea

> Find new axes (directions) such that:
>
> - First axis captures **maximum variance**
> - Second axis captures next highest variance
> - And so on...

---

## What is Variance?

Variance measures how **spread out** the data is.

:contentReference[oaicite:0]{index=0}

- High variance → more information
- Low variance → less useful

---

## PCA Transformation

Original data → New coordinate system

:contentReference[oaicite:1]{index=1}

Where:

- \( X \) = original data
- \( W \) = principal components (directions)
- \( Z \) = transformed data

---

# Step-by-Step Working of PCA

## Step 1: Standardize the Data

- Mean = 0
- Variance = 1

> Important because PCA is sensitive to scale

---

## Step 2: Compute Mean Vector

:contentReference[oaicite:2]{index=2}

---

## Step 3: Compute Covariance Matrix

:contentReference[oaicite:3]{index=3}

---

### What is Covariance?

- Measures how features vary together
- Positive → move together
- Negative → move oppositely

---

## Step 4: Compute Eigenvalues and Eigenvectors

:contentReference[oaicite:4]{index=4}

Where:

- \( v \) = eigenvector (direction)
- \( \lambda \) = eigenvalue (importance)

---

## Step 5: Sort Components

- Sort eigenvectors by **largest eigenvalues**
- Highest eigenvalue → most important direction

---

## Step 6: Select Top K Components

- Choose top K eigenvectors
- Reduce dimensionality

---

## Step 7: Transform Data

:contentReference[oaicite:5]{index=5}

---

# Intuition Behind PCA

- Finds directions where data varies the most
- Projects data onto those directions
- Removes less important dimensions

---

# Geometric Interpretation

- PCA rotates coordinate system
- Aligns axes with data spread
- Projects data onto new axes

---

# Explained Variance

Each principal component explains some variance:

:contentReference[oaicite:6]{index=6}

---

## Choosing Number of Components

- Keep components that explain **~95% variance**
- Use scree plot (variance vs components)

---

# Example (Simple)

2D data:

- Highly correlated features

PCA:

- Combines them into 1 main component
- Reduces dimension from 2 → 1

---

# Training vs Inference

## Training Phase

- Compute mean
- Compute covariance matrix
- Compute eigenvectors

---

## Inference Phase

- Subtract mean
- Multiply with principal components

```c
Z = X * W;
```

## Advantages of PCA

- Reduces dimensionality
- Removes redundancy
- Improves model performance
- Helps visualization

## Limitations of PCA

- Loses interpretability
- Linear method (cannot capture non-linear patterns)
- Sensitive to scaling
- Affected by outliers

## PCA vs Feature Selection

| Feature          | PCA                | Feature Selection |
| ---------------- | ------------------ | ----------------- |
| Approach         | Transform features | Select subset     |
| Output           | New features       | Original features |
| Interpretability | Low                | High              |
