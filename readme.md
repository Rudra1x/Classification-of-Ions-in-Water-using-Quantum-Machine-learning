### Problem
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


Variant,Qubits,Layers,Encoder Hidden
v6_2,6,2,48
v7_3,7,3,48
v8_3,8,3,64
```





⚛️ Quantum DetailsBackend: 
-PennyLane default.qubit (simulator).
-Input encoding: AngleEmbedding with angles = tanh(encoder_output) * π (ensures bounds).
-Ansatz: StronglyEntanglingLayers(weights) with shape (n_layers, n_qubits, 3).
-QNode returns: [⟨Z_0⟩, ..., ⟨Z_n-1⟩] used as classical features for the head.
🖥️ Classical HeadDense layers: Linear(n_qubits, hidden) $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear(hidden, 4).
