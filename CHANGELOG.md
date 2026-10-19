# Project Change Log

## [v2.0.0] - Ensemble, Specialist Networks & Symmetry Augmentation

Validation cost (log2 MSE): **0.620 → 0.594**. Notebook: `Project1ver2.ipynb`.

### Analyzed
- **Per-(k, m) evaluation**: the three groups with `m = n − k` cause about 80% of the error. There the m-height is a ratio of sub-matrix determinants, so it jumps sharply near singular sub-matrices.
- **Under-fitting, not over-fitting**: in v1 the training and validation losses were almost equal.

### Updated
- **Loss = metric**: log-cosh replaced by MSE on `log2(y)`, which is the grading cost.
- **Dropout removed**, because the model was under-fitting.
- **Ensemble of 3 networks** (different seeds), averaged with a Keras `Average` layer: 0.620 → 0.609.

### Added
- **Specialist network per (k, m)**: its input is that group's `P` only, without padding.
- **Symmetry data augmentation (training only)**: row/column permutations and sign flips of `P` leave the m-height exactly unchanged. This removed over-fitting: `(6,2)` went from 0.130 to 0.029.
- **Mixture-of-experts routing** (`routing.json`): per (k, m), the ensemble, the specialist or their average, chosen on the validation set.
- **Safe post-processing**: NaN/inf guard, and `1 ≤ ŷ < ∞` enforced by an assertion.

### Tried and rejected
- `layers.Multiply` (gated) layers with 1.5M extra LP-generated samples for the hard groups: 0.669.
- Symmetry augmentation for the shared MLP: worse, because that model was under-fitting.

---

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
