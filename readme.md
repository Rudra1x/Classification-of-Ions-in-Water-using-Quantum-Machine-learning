#Problem
Classify which ion (C₀, C₁, C₂, C₃) is present in an aqueous sample using three photonic-sensor features: wavelength, propagation constant (beta), electric field fraction (frac_campo).

### Solution
A hybrid pipeline:
1.  **Classical encoder** (pretrained)
2.  **Variational Quantum Circuit** (PennyLane `StronglyEntanglingLayers`) wrapped with `qml.qnn.TorchLayer`
3.  **Classical head** $\rightarrow$ softmax per model

Train three hybrid variants (different qubits / layers), K-Fold each, produce OOF probabilities, then stack with a logistic meta-learner.

### Final OOF (Stacked)
**67.76%** — best submission artifact.

---

# 📚 Table of contents

* [Summary & Motivation](#-summary--motivation)
* [Dataset & Features](#-dataset--features)
* [Preprocessing](#-preprocessing)
* [Model Architecture & Quantum Component](#-model-architecture--quantum-component)
* [Training Strategy](#-training-strategy)
* [Results & Analysis](#-results--analysis)
* [Files & Deliverables](#-files--deliverables)
* [Reproduce / Run (Commands)](#-reproduce--run-commands)
* [Practical Tips for Judges & Presentation](#-practical-tips-for-judges--presentation)
* [Limitations & Future Work](#-limitations--future-work)
* [License](#%E2%9A%96%EF%B8%8F-license)

---

## 💡 Summary & Motivation

Contaminated water is a global risk — fast, reliable detection is crucial. The photonic sensor produces three physics-grounded features that change with ion presence. We propose a hybrid QML approach: classical preprocessing + small classical encoder $\rightarrow$ quantum variational circuit $\rightarrow$ classical head.

To maximize robustness we:
1.  Pretrain the classical encoder (cheap).
2.  Run several hybrid variants (different expressivity).
3.  Ensemble and stack their predicted probabilities with a lightweight logistic meta-learner.

### Why hybrid / quantum?

* **Expressivity:** Quantum circuit offers a compact, expressive nonlinear map from encoded classical features to expectation-value space.
* **Rich Boundaries:** In low-dimensional input settings, hybrid models can provide richer decision boundaries with fewer classical parameters.
* **Stability:** Pretraining the encoder reduces the quantum training burden and stabilizes optimization.

---

## 📊 Dataset & Features

* **Files provided:** `C0_dataset.csv`, `C1_dataset.csv`, `C2_dataset.csv`, `C3_dataset.csv`.
* **Total samples:** 48,000 (12,000 per class).
* **Features (per sample):**
    * `lambda` — wavelength (real-valued)
    * `beta` — propagation constant
    * `frac_campo` — electric field fraction overlapping water
* **Label:** integer 0..3 (C₀..C₃)

---

## ⚙️ Preprocessing

* Concatenate the four CSVs into one dataset.
* `StandardScaler` per fold (fit on train split, apply to val/test).
* No feature generation beyond scaling — the 3 physics features are already meaningful and low-dimensional (we keep PCA out to let encoder learn).
* Class balancing approach: nearly-balanced dataset, but we add gentle weight for class 3 in loss when needed.

---

## 🧠 Model Architecture & Quantum Component

```mermaid
flowchart LR
  A[Raw CSVs] --> B[Concatenate]
  B --> C[StandardScaler]
  C --> D[Classical Encoder (Dense)]
  D --> E[Angle Encoding: tanh(latent)*π]
  E --> F[Variational QC: StronglyEntanglingLayers]
  F --> G[Classical head]
  G --> H[Softmax per-variant]
  H --> I[OOF probs -> Stack (Logistic) -> Final label]
```


### ⚛️ Quantum Details
* **Backend:** PennyLane `default.qubit` (simulator).
* **Input encoding:** AngleEmbedding with angles = `tanh(encoder_output) * π` (ensures bounds).
* **Ansatz:** `StronglyEntanglingLayers(weights)` with shape `(n_layers, n_qubits, 3)`.
* **QNode returns:** `[⟨Z_0⟩, ..., ⟨Z_n-1⟩]` used as classical features for the head.

### 🖥️ Classical Head
* **Dense layers:** `Linear(n_qubits, hidden)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout` $\rightarrow$ `Linear(hidden, 4)`.

---

## 🚀 Training Strategy
* **K-Fold:** `KFOLD = 3` (stratified) to get out-of-fold (OOF) probabilities for stacking.
* **Pretrain:** Pretrain encoder classically (with a small head) for a few epochs to warm-start the encoder.
* **Hybrid training:** Train full model (encoder + q-layer + head) with `FocalLoss` (gamma ~ 1.5) or `CrossEntropy` with slight class weights as needed.
* **Optimizers:**
    * Pretrain: `Adam` (1e-3)
    * Hybrid: `Adam` (5e-4)
* **Schedulers:** `ReduceLROnPlateau` on validation accuracy.
* **Early stopping:** Patience ~6–7 epochs.
* **Ensembling:** Soft-average of variant probabilities $\rightarrow$ stacked meta-model (`LogisticRegression` on OOF probabilities).
* **Final submission:** Use stacked predictions (meta model), saved reproducibly.

---

## 📈 Results (Final Artifacts)

### Per-Variant OOF Accuracy
* **v6\_2:** 64.88%
* **v7\_3:** 66.49%
* **v8\_3:** 65.95%

### Ensembling
* **Soft-average ensemble OOF accuracy:** 66.44%
* **Stacked (logistic) OOF accuracy:** **67.76%** $\leftarrow$ **Final Choice**

### Key Metrics (Stacked)

**Overall Accuracy: 67.76%**

**Per-Class Metrics (Stacked):**
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| C0 | 0.5967 | 0.4274 | 0.4981 |
| C1 | 0.6039 | 0.9182 | 0.7286 |
| C2 | 0.9770 | 0.9823 | 0.9797 |
| C3 | 0.5048 | 0.3826 | 0.4353 |

> **Interpretation:** High accuracy for C2 and C1 (recall is very strong for C1 and C2). C0 and C3 need more attention — we mitigated this by stacking, but further improvements are possible (see Limitations & Future Work).

---


python train_hybrid_variants.py \
  --data-files C0_dataset.csv C1_dataset.csv C2_dataset.csv C3_dataset.csv \
  --kf 3 --batch 128 --pretrain-epochs 8 --hybrid-epochs 20
Inference (Produces submission.csv)
Bash

python inference.py --models-dir ./models --meta meta_logistic.pkl --out submission.csv
Quick Test / Micro-benchmark
Run the small snippet in the training environment to measure per-batch cost and scale up estimates.




🔬 Limitations & Future Work
(How to push beyond 67.8%)

More expressive encoding: Try small quantum kernels (QSVM) or amplitude encoding if you can increase qubit resources safely.

Data augmentation: If realistic sensor-noise models exist, augment training data.

Class-specific fine-tuning: Per-class specialized heads (or cost-sensitive training) to boost C0/C3.

Temperature scaling / calibration: Calibrate stacked probabilities with a validation set.

Use a faster quantum backend: Use PennyLane-Lightning or GPU-accelerated simulators to explore 10+ qubits / deeper layers.

Ensemble diversity: Add more variant classes (different encoders, quantum ansatz types, classical baselines) — stacking benefits from diversity.

We fused sensor physics + classical representation learning + quantum circuit expressivity and used stacking to boost robustness — resulting in a reproducible hybrid pipeline achieving 67.8% OOF accuracy on a 4-way ion classification task.

📜 Appendix — Quick Inference Pseudo-code
Python

# load meta (joblib)
meta = joblib.load("meta_logistic.pkl")

probs_per_model = []

for model_file, scaler_file in models_list:
    scaler = joblib.load(scaler_file)
    X_scaled = scaler.transform(X_test)
    
    model = HybridQNN(...)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    probs = model.predict_proba(X_scaled)  # via batched torch inference -> softmax
    probs_per_model.append(probs)

# concat along axis=1 for meta features
meta_features = np.concatenate(probs_per_model, axis=1)

final_preds = meta.predict(meta_features)
👥 Contact & Credits

Primary author: Ruraksh Sharma

Code & artifacts: Included in repository root.

⚖️ License
MIT — feel free to reuse and adapt. Please attribute if used in publications.
* `ensemble_avg_probs.npy`, `stacked_oof_probs.npy` — Ensemble outputs
* `README.md` — This file
* `ablation.pdf` — 1-page figure comparing single-variant OOF vs soft-average vs stacked (export from notebook)

---

## 🔄 Reproduce — Environment & Commands

### Environment Setup
(Recommended conda env)
```bash
conda create -n qml-ion python=3.10 -y
conda activate qml-ion
pip install numpy pandas scikit-learn torch pennylane matplotlib joblib tqdm
# (If you have a faster PennyLane plugin, install it here, e.g., pennylane-lightning)

