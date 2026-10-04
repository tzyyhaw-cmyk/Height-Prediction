# Deep Neural Network Height-Prediction
This repository contains a pure **Deep Neural Network (DNN)** implementation designed to predict height parameters (`mHeights`) from multi-dimensional input vectors ($n, k, m, P$). The project strictly focuses on deep learning principles, target scaling optimization, and smooth gradient convergence.

---

## Project Overview

### 📍 Situation
In physical modeling and data-driven estimation, predicting target parameters like `mHeights` ($y$) from multi-dimensional numerical inputs ($n, k, m, P$) presents substantial non-linear regression challenges. The raw target variable exhibits a wide dynamic range and significant scale variance, which frequently leads to unstable gradient updates, slow model convergence, and biased loss metrics in standard deep learning pipelines.

### 🎯 Task
The primary goal was to construct an end-to-end regression model utilizing **strictly pure Deep Neural Network (DNN)** architectures—avoiding custom multi-branch fusion or feature-engineering heuristics. Specific objectives included:
1. Stabilizing gradient flow across large numerical target ranges.
2. Preventing vanishing/exploding gradients and dying activation units.
3. Achieving robust generalization without overfitting.
4. Restoring inference outputs back to the original unit scale ($y$) for realistic performance evaluation.

### 🛠️ Action
To address these challenges, the following core strategies were implemented:
* **Target Log2 Transformation**: Applied $z = \log_2(y)$ transformation on the target height values to compress scale variance, yielding a well-behaved target distribution for MSE loss optimization.
* **ELU Activation Units**: Configured **ELU (Exponential Linear Unit)** activation functions across all hidden dense layers. ELU provides smooth negative values that bring unit activations closer to zero mean, speeding up learning and overcoming the "dying ReLU" problem.
* **Pure DNN Architecture & Regularization**: Designed a clean, fully connected Multi-Layer Perceptron (MLP) enhanced with `Dropout` layers to enforce feature robustness and prevent overfitting.
* **Optimization Pipeline**: Trained using the Adam optimizer alongside `EarlyStopping` (patience=15) and `ModelCheckpoint` callbacks to automatically save the optimal model weight state (`best_model.keras`).
* **Original Scale Post-Processing**: Integrated an inverse transformation ($\hat{y} = 2^{\hat{z}}$) during evaluation and test inference to calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) directly on original $y$-scale values.

### 📊 Result
* **Optimal Convergence**: The pure DNN model achieved clean, smooth convergence around Epochs 50–70.
* **Zero Overfitting**: Training Loss and Validation Loss aligned tightly (~0.60–0.62 in $\log_2$ space), demonstrating exceptional generalization on unseen validation data.
* **Representation Capacity**: Confirmed that the deep architecture fully extracted available signals from the feature space while maintaining a highly efficient and pure neural network structure.
<img width="937" height="576" alt="loss_curve" src="https://github.com/user-attachments/assets/b0c8bcdc-d2ab-4ea6-8336-6030c63d1545" />

---

## Key Highlights & Pipeline

```text
[Input Vectors (n, k, m, P)]
           │
           ▼
[Pure Deep Neural Network (Dense + ELU + Dropout)]
           │
           ▼
 [Log2 Target Output (z_hat)] ──► [Inverse Transform (2^z_hat)] ──► [Final Height Prediction (y_hat)]
