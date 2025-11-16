Problem: classify which ion (C₀, C₁, C₂, C₃) is present in an aqueous sample using three photonic-sensor features: wavelength, propagation constant (beta), electric field fraction (frac_campo).

Solution: a hybrid pipeline:

Classical encoder (pretrained)

Variational Quantum Circuit (PennyLane StronglyEntanglingLayers) wrapped with qml.qnn.TorchLayer

Classical head → softmax per model
Train three hybrid variants (different qubits / layers), K-Fold each, produce OOF probabilities, then stack with a logistic meta-learner.
Final OOF (stacked): 67.76% — best submission artifact.

📚 Table of contents

Summary & Motivation

Dataset & Features

Preprocessing

Model Architecture & Quantum Component

Training Strategy

Results & Analysis

Files & Deliverables

Reproduce / Run (Commands)

Practical Tips for Judges & Presentation

Limitations & Future Work

License

# Summary & Motivation

Contaminated water is a global risk — fast, reliable detection is crucial. The photonic sensor produces three physics-grounded features that change with ion presence. We propose a hybrid QML approach: classical preprocessing + small classical encoder → quantum variational circuit → classical head. To maximize robustness we: (1) pretrain the classical encoder (cheap), (2) run several hybrid variants (different expressivity), (3) ensemble and stack their predicted probabilities with a lightweight logistic meta-learner.

Why hybrid / quantum?

Quantum circuit offers a compact, expressive nonlinear map from encoded classical features to expectation-value space. In low-dimensional input settings, hybrid models can provide richer decision boundaries with fewer classical parameters.

Pretraining encoder reduces quantum training burden and stabilizes optimization.

# Dataset & Features

Files provided: C0_dataset.csv, C1_dataset.csv, C2_dataset.csv, C3_dataset.csv.
Total samples: 48,000 (12,000 per class).
Features (per sample):

lambda — wavelength (real-valued)

beta — propagation constant

frac_campo — electric field fraction overlapping water

Label: integer 0..3 (C₀..C₃)

# Preprocessing

Concatenate the four CSVs into one dataset.

StandardScaler per fold (fit on train split, apply to val/test).

No feature generation beyond scaling — the 3 physics features are already meaningful and low-dimensional (we keep PCA out to let encoder learn).

Class balancing approach: nearly-balanced dataset, but we add gentle weight for class 3 in loss when needed.

# Model architecture & quantum component (diagram)
flowchart LR
  A[Raw CSVs] --> B[Concatenate]
  B --> C[StandardScaler]
  C --> D[Classical Encoder (Dense)]
  D --> E[Angle Encoding: tanh(latent)*π]
  E --> F[Variational QC: StronglyEntanglingLayers]
  F --> G[Classical head]
  G --> H[Softmax per-variant]
  H --> I[OOF probs -> Stack (Logistic) -> Final label]


Variants trained

v6_2: 6 qubits, 2 layers, encoder_hidden=48

v7_3: 7 qubits, 3 layers, encoder_hidden=48

v8_3: 8 qubits, 3 layers, encoder_hidden=64

Quantum details

Backend: PennyLane default.qubit (simulator).

Input encoding: AngleEmbedding with angles = tanh(encoder_output) * π (ensures bounds).

Ansatz: StronglyEntanglingLayers(weights) with shape (n_layers, n_qubits, 3).

QNode returns [⟨Z_0⟩, ..., ⟨Z_n-1⟩] used as classical features for head.

Classical head

Dense layers: Linear(n_qubits, hidden) -> ReLU -> Dropout -> Linear(hidden, 4).

# Training strategy

K-Fold: KFOLD = 3 (stratified) to get out-of-fold (OOF) probabilities for stacking.

Pretrain encoder classically (small head) for a few epochs to warm-start the encoder.

Hybrid training: train full model (encoder + q-layer + head) with FocalLoss (gamma ~ 1.5) or CrossEntropy with slight class weights as needed.

Optimizers:

Pretrain: Adam 1e-3

Hybrid: Adam 5e-4

Schedulers: ReduceLROnPlateau on validation accuracy.

Early stopping: patience ~6–7 epochs.

Ensembling: soft-average of variant probabilities → stacked meta (LogisticRegression on OOF probabilities).

Final submission: use stacked predictions (meta model), saved reproducibly.

# Results (final artifacts)

Per-variant OOF accuracy:

v6_2: 64.88%

v7_3: 66.49%

v8_3: 65.95%

Ensembling

Soft-average ensemble OOF accuracy: 66.44%

Stacked (logistic) OOF accuracy: 67.76% ← final choice

Key metrics (stacked)

Overall accuracy: 67.76%
Per-class (stacked):
  C0: precision 0.5967, recall 0.4274, f1 0.4981
  C1: precision 0.6039, recall 0.9182, f1 0.7286
  C2: precision 0.9770, recall 0.9823, f1 0.9797
  C3: precision 0.5048, recall 0.3826, f1 0.4353


Interpretation: high accuracy for C2 and C1 (recall very strong for C1 and C2). C0 and C3 need more attention — we mitigated this by stacking but further improvements are possible (see Limitations & Future Work).

# Files & deliverables (what we provide)

train_hybrid_variants.py — full training pipeline (pretrain encoder, hybrid training, KFOLD, save models, produce OOF probs)

inference.py — deterministic inference script that:

loads scalers + per-fold .pth model files,

produces per-variant probabilities,

concatenates features in the order used by the meta model,

applies meta_logistic.pkl to produce final labels and probabilities,

writes submission.csv.

meta_logistic.pkl — logistic stacking model (trained on OOF features)

v6_2_fold1.pth, ... v8_3_fold3.pth — saved PyTorch hybrid weights

scaler_v*_fold*.pkl — saved StandardScalers per fold

oof_probs_variants.npy — OOF probabilities for variants

ensemble_avg_probs.npy, stacked_oof_probs.npy — ensemble outputs

README.md — this file

ablation.pdf — 1-page figure comparing single-variant OOF vs soft-average vs stacked (export from notebook)

# Reproduce — environment & commands
Recommended conda env (copy/paste)
conda create -n qml-ion python=3.10 -y
conda activate qml-ion
pip install numpy pandas scikit-learn torch pennylane matplotlib joblib tqdm
# (If you have a faster PennyLane plugin, install it here, e.g., pennylane-lightning)

Train (full pipeline)
python train_hybrid_variants.py \
  --data-files C0_dataset.csv C1_dataset.csv C2_dataset.csv C3_dataset.csv \
  --kf 3 --batch 128 --pretrain-epochs 8 --hybrid-epochs 20

Inference (produces submission.csv)
python inference.py --models-dir ./models --meta meta_logistic.pkl --out submission.csv

Quick test / micro-benchmark (single batch forward+back)

Run the small snippet in the training environment to measure per-batch cost and scale up estimates.

# How to prepare submission artifacts (recommended)

models.zip — include all v*_fold*.pth + scaler_*.pkl + meta_logistic.pkl.

submission.csv — format: id,pred_label,prob_C0,prob_C1,prob_C2,prob_C3.

report.pdf (2 pages): short methods, architecture diagram, metrics, confusion matrix.

presentation.mp4 (3–5 min): team members present approach, encoding choice, and results.



# Practical tips to impress judges

Show reproducibility: run inference.py end to end in a minute; show submission.csv matches OOF statistics.

Explain physically: connect each feature to sensor physics — judges love domain grounding.

Visuals: include a confusion-matrix heatmap, per-class ROC/f1 bar chart, and a tiny animation showing how angle encoding maps data to quantum states (simple 2D illustration).

Ablation: show single-variant vs ensemble vs stacked improvement clearly (1 slide).

Resource honesty: mention that QNode uses default.qubit (simulator) and describe how this would map to a real quantum backend or future GPU-accelerated simulators.

Limitations: be honest about classes C0/C3 tradeoffs and propose immediate fixes (below).

# Limitations & future work (how to push beyond 67.8%)

More expressive encoding

Try small quantum kernels (QSVM) or amplitude encoding if you can increase qubit resources safely.

Data augmentation

If realistic sensor-noise models exist, augment training data.

Class-specific fine-tuning

Per-class specialized heads (or cost-sensitive training) to boost C0/C3.

Temperature scaling / calibration

Calibrate stacked probabilities with a validation set.

Use a faster quantum backend for larger circuits

PennyLane-Lightning or GPU-accelerated simulators to explore 10+ qubits / deeper layers.

Ensemble diversity

Add more variant classes (different encoders, quantum ansatz types, classical baselines) — stacking benefits from diversity.

# Judge-friendly one-liner (for your slide)

We fused sensor physics + classical representation learning + quantum circuit expressivity and used stacking to boost robustness — resulting in a reproducible hybrid pipeline achieving 67.8% OOF accuracy on a 4-way ion classification task.

# Appendix — Quick inference pseudo-code
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

# Contact & credits

Team: Quantum Sensing Squad
Primary author: Ruraksh Sharma
Code & artifacts: included in repository root.


# License

MIT — feel free to reuse and adapt. Please attribute if used in publications.
