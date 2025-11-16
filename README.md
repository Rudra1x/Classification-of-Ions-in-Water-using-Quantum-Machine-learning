<div align="center">

# QWater-Classify 

### A Hybrid Quantum-Classical Pipeline for Classifying Water Contaminants

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red?style=for-the-badge&logo=pytorch)
PennyLane
![Accuracy](https://img.shields.io/badge/Final%20Accuracy-67.76%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**This project implements a state-of-the-art Hybrid Quantum-Classical Neural Network (HQ-NN) to identify four different types of contaminant ions ($C_0, C_1, C_2, C_3$) in water, using experimental data from photonic sensors.**


</div>

---

## 📍 Table of Contents

* [About the Challenge](#-about-the-challenge)
* [Key Features](#-key-features)
* [The Hybrid Quantum-Classical Solution](#-the-hybrid-quantum-classical-solution)
* [Architecture Deep-Dive](#-architecture-deep-dive)
* [🚀 Performance Highlights](#-performance-highlights)
* [Getting Started](#-getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Training the Model](#training-the-model)
* [Future Improvements](#-future-improvements)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

---

## Problem
## 🎯 About the Challenge

The goal is to solve a 4-class classification problem: **identifying which of four contaminant ions ($C_0, C_1, C_2, C_3$) is present in a water sample**.

[cite_start]The only available data comes from a photonic sensor, which provides three continuous physical parameters for each sample:
1.  **Wavelength ($\lambda$)** 
2.  **Propagation Constant ($\beta$)**
3.  **Electric Field Fraction**

The relationships between these features and the ion type are complex and non-linear, making this an ideal testbed for Quantum Machine Learning (QML) models [cite: 15, 16] that can explore high-dimensional spaces to find decision boundaries.

---

## ✨ Key Features

* **Hybrid Quantum-Classical Model:** Uses **PyTorch** for classical pre- and post-processing and **PennyLane** for the quantum core.
* **Learnable Encoding:** Employs a classical encoder to learn the best way to represent 3 features in an $N_q$-dimensional quantum space.
* **Robust Ensemble:** The final solution is a **stacked ensemble** of three HQ-NN variants (using 6, 7, and 8 qubits), combined with a Logistic Regression meta-model for a final, robust prediction.
* **Advanced Training:** Uses a 3-fold stratified cross-validation scheme, **two-stage pretraining**, and **Focal Loss** to focus on hard-to-classify examples.

---

## 🔬 The Quantum Solution

We developed a Hybrid Quantum Neural Network (HQ-NN) designed to leverage the best of both worlds.


1.  **Classical Encoder:** A classical feed-forward network (built in PyTorch) takes the 3 raw sensor features and "encodes" them into a higher-dimensional latent vector. This step is trainable, meaning the model *learns* the most effective way to prepare the data for the quantum circuit.
2.  **Variational Quantum Circuit (VQC):** The latent vector is used to parameterize a VQC (built in PennyLane). We use the powerful `qml.StronglyEntanglingLayers` ansatz to process the information. The circuit's output is a vector of expectation values ($\langle\sigma_{Z}\rangle$) from each qubit.
3.  **Classical Head:** A final classical network (PyTorch) receives the measurement vector from the VQC and performs the final 4-class classification.

To maximize performance, we trained an **ensemble** of three models (v6_2, v7_3, v8_3) with different qubit counts and layers and stacked their predictions.

---

## 🔧 Architecture Deep-Dive

This diagram shows the flow of a single data point through one of the HQ-NN variants.

> **[ 3 Features ]** $\rightarrow$ `Classical Encoder` $\rightarrow$ **[ $N_q$ Latent Features ]** $\rightarrow$ `VQC (AngleEmbedding + StronglyEntanglingLayers)` $\rightarrow$ **[ $N_q$ Expectation Values ]** $\rightarrow$ `Classical Head` $\rightarrow$ **[ 4-Class Logits ]**

### 1. Classical Encoder
* **Input:** 3 features
* **Layers:** Linear (3 $\rightarrow$ H) $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear (H $\rightarrow$ $N_q$).
* **Output:** $N_q$-dimensional latent vector (where $N_q$ is 6, 7, or 8).

### 2. Variational Quantum Circuit (VQC)
* **Encoding:** The $N_q$ latent features are bounded to $(-\pi, \pi)$ using a `tanh` function and loaded into the VQC using `qml.AngleEmbedding` (as $R_Y$ rotations).
* **Processing:** The state is processed by `qml.StronglyEntanglingLayers` (2 or 3 layers).
* **Measurement:** We measure the Pauli-Z expectation value for each of the $N_q$ qubits.

### 3. Classical Head
* **Input:** $N_q$ expectation values
* **Layers:** Linear ($N_q$ $\rightarrow$ 24) $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear (24 $\rightarrow$ 4) 
* **Output:** Raw logits for the 4 contaminant classes ($C_0, C_1, C_2, C_3$).

---

## 🚀 Performance Highlights

The final **Stacked Ensemble** model achieves a robust **Out-of-Fold Accuracy of 67.76%**.

### Per-Class Performance (F1-Score)

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Ion $C_0$ | 0.597 | 0.427 | 0.498 |
| Ion $C_1$ | 0.604 | 0.918 | 0.729 |
| Ion $C_2$ | 0.977 | 0.982 | **0.980 |
| Ion $C_3$ | 0.505 | 0.383 | 0.435 |
| **Weighted Avg** | 0.671** |0.678 | 0.660 |
*(Data from Table 2 )*

### Key Takeaways
*  **Excellent at $C_2$:** The model identifies Ion $C_2$ almost perfectly, with an F1-score of ~98%.
*  **Strong at $C_1$:** The model is also very effective at finding Ion $C_1$, with a high recall of ~92% (meaning it rarely misses it).
*  **Primary Weakness:** The model struggles to distinguish between Ion $C_0$ and Ion $C_3$[cite: 91, 99]. [cite_start]The confusion matrix shows that these two classes are frequently misclassified as each other, suggesting their sensor profiles are highly similar.

---

## 💻 Getting Started

### Prerequisites

This project requires Python 3.8+ and the libraries listed in `requirements.txt`.
* [PyTorch](https://pytorch.org/get-started/locally/)
* [PennyLane](https://pennylane.ai/install/)
* Scikit-learn, Pandas, Numpy

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/QWater-Classify.git](https://github.com/your-username/QWater-Classify.git)
    cd QWater-Classify
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Training the Model

1.  **Run the main training script:**
    ```bash
    # This will run the full 3-fold CV for all 3 model variants
    python train.py
    ```

2.  **Generate Predictions:**
    ```bash
    # This script will use the trained models to make predictions on new data
    python predict.py --input data/unseen_data.csv --output data/predictions.csv
    ```
*(Note: These file names are illustrative examples; adapt to your project structure.)*

---

## 📈 Future Improvements

While this model provides a strong baseline, there are several avenues for improvement:

1.  **Quantum Kernel Methods:** For this low-dimensional (3-feature) input, a Quantum Kernel (QK) approach with a classical SVM could be highly effective.
2.  **Data Re-uploading:** Increase model expressivity without increasing qubits by re-uploading the input features between ansatz layers.
3.  **Advanced Feature Engineering:** To solve the $C_0$/$C_3$ confusion, new classical features (e.g., ratios, polynomial terms) could be created and fed to the encoder.
4.  **Custom Ansatz Design:** Explore problem-specific or hardware-efficient ansatze instead of the generic `StronglyEntanglingLayers`.

---

---

## 🙏 Acknowledgments

* This project is an implementation of the solution detailed in the technical report: "*A Quantum Approach for the Classification of Contaminant Ions in Water*" by Rudraksh Sharma (Nov 16, 2025)
* Built with ❤️ using **PennyLane** and **PyTorch**.

</div>
