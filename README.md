# Challenge 3 — Quantum Machine Learning: Classification of Ions in Water

1. Overview

Water contamination detection is a high-impact problem requiring fast, accurate, and intelligent sensing. In this challenge, participants will combine experimental data from a photonic sensor with Quantum Machine Learning (QML) to classify contaminant ions dissolved in water.

Your mission:

👉 Build a quantum or hybrid quantum–classical model capable of identifying which ion is present in a given solution sample.

2. Scenario

A photonic sensor has been developed to analyze aqueous solutions. When the sensor interacts with specific contaminant ions in water, its optical response changes depending on the ion present.

You are provided with experimental data obtained from this sensor for four different contaminant ions:
*	C_0
*	C_1
*	C_2
*	C_3

Your goal is to use QML techniques to automatically classify which ion is present in each sample solely based on the sensor’s optical response.

3. Main Task

Design, implement, and train a Quantum Machine Learning model that:
*	Takes as input three physical parameters measured by the photonic sensor.
*	Outputs a prediction of which ion class (C_0,C_1,C_2,C_3) corresponds to each sample.
*	Achieves as high a classification accuracy as possible on unseen (test) data.

You are free to choose the QML approach (e.g., variational quantum circuits, quantum kernels, hybrid quantum-classical models, etc.) as long as it genuinely uses a quantum component.

4. Input Data

Each sample in the dataset corresponds to a particular optical measurement of the sensor in contact with an aqueous solution containing one ion type. For each sample, you are given the following three features:
1.	Wavelength
    *	Description: Wavelength of the light used in the measurement.
    *	Role: Input parameter that controls how the sensor is probed.
2.	Propagation Constant
    *	Description: The effective wavenumber associated with the mode propagating in the system.
    *	Role: Fundamental dispersion parameter that reflects how light propagates in the sensing region and is influenced by the ion present.
3.	Electric Field Fraction
    *	Description: Fraction of the electric field energy overlapping with the water sample region.
    *	Role: Indicates the sensor’s sensitivity to changes in the medium (i.e., how strongly the optical field interacts with the aqueous solution).

Each data point is labeled with the corresponding ion class: C_0,C_1,C_2,or C_3. This mean-Each sample includes one of the ion labels: C₀, C₁, C₂, C₃.

You must use these three parameters as the input features for your QML model.

5. Expected Outputs

Your solution should produce:
1.	A trained QML model capable of multi-class classification of the four ions.
2.	Predictions on a held-out test set, with:
3.	A technical report or notebook explaining:
*	Data preprocessing and feature scaling/normalization (if any)
*	Encoding strategy from classical features to quantum states (e.g., angle encoding, amplitude encoding, etc.)
*	Quantum model architecture (ansatz, number of qubits, number of layers)
*	Training strategy and optimization method
*	Results, limitations, and possible improvements

4.	Use of quantum resources
    *	Clear description of the quantum component
	  * Justification of why a quantum (or hybrid) model is relevant or promising for this type of data
5.	Scientific and technical soundness
    *	Physically reasonable data handling
    *	Thoughtful choices for encoding and model design
6.	Clarity and Reproducibility
    *	Clean, well-documented code/notebook
    *	Clear explanation of methods and results

📄 Deliverable 1 — QML Model + Code (Required)
Submit code or a notebook that implements the full QML pipeline:
*	Data preprocessing and normalization
*	Classical-to-quantum encoding strategy
*	Quantum circuit / model architecture
*	Training loop
*	Metrics and evaluation
*	Visualizations (optional but recommended)

📄 Deliverable 2 — Technical Report (Required)

A report (PDF or Markdown) that explains:

*  Understanding of the problem
*	Your model architecture
*	Classical-to-quantum encoding method
*	Training process and optimization details
*	Key results
*	Confusion matrix + accuracy + at least one extra metric (F1, precision, recall)
*	Limitations and possible improvements

This document must be clear, detailed, and reproducible.

🎥 Deliverable 3 — Team Presentation Video (Required)

*	Length: 3–5 minutes
*	All team members must appear
*	Summarize:
    *	Your model
    *	Encoding strategy
    * Key results

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=REPLACEWITHGITHUBURL)
