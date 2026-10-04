# Project Change Log

## [v1.2.0] - Activation Function & Pure DNN Optimization

### Updated
- **Transitioned to ELU Activation**: Replaced standard activations with **ELU (Exponential Linear Unit)** across the hidden layers. ELU speeds up learning and mitigates the dying ReLU problem by outputting negative values to bring the mean unit activation closer to zero.
- **Strict Pure DNN Architecture**: Standardized the model strictly on a fully connected **Deep Neural Network (DNN / MLP)** architecture. Avoided hybrid, multi-branch, or custom fusion layers to keep the network end-to-end, lightweight, and pure.

### Analyzed
- **Loss Convergence & Capacity Limit**: Training and validation losses converge stably and align tightly at ~0.60–0.62 around epochs 50–70. The lack of overfitting confirms that the pure DNN setup effectively captures all available feature signals within the current representation.

---

## [v1.1.0] - Post-Processing & Original Scale Evaluation

### Added
- **Prediction Inverse Transformation ($\hat{y} = 2^{\hat{z}}$)**: Implemented logarithmic inverse mapping during inference to restore predictions back to original height units (`mHeights`).
- **Original Scale Evaluation**: Integrated evaluation scripts to measure real-world Performance (MSE and MAE on $y$) alongside log2-space loss.
- **Inference Pipeline**: Built test script to load test dataset (`CSCE-636-Project-1-Test-n_k_m_P`) and export final outputs.

---

## [v1.0.0] - Baseline Pipeline & Target Scaling

### Added
- **Target Log2 Normalization**: Applied $z = \log_2(y)$ transformation on target height variable (`mHeights`) to stabilize optimization gradients and handle wide numerical dynamic ranges.
- **DNN Training Environment**: Established baseline training setup using Adam optimizer, MSE loss, `EarlyStopping` (patience=15), and `ModelCheckpoint` to continuously capture `best_model.keras`.
