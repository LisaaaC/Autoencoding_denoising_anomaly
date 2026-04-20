# Autoencoder for ECG Anomaly Detection using TensorFlow

This project implements an autoencoder in TensorFlow for anomaly detection on ECG time-series data. The model learns to reconstruct normal heartbeat patterns and identifies anomalous signals based on reconstruction error.

The work is inspired by TensorFlow's official **Intro to Autoencoders** tutorial, which introduces:
- a basic autoencoder,
- image denoising with autoencoders,
- anomaly detection using the ECG5000 dataset.

According to the tutorial, autoencoders are neural networks trained to copy their input to their output while learning a lower-dimensional latent representation that minimizes reconstruction error. In the ECG example, the model is trained only on normal rhythms, and abnormal rhythms are detected when reconstruction loss exceeds a chosen threshold.

## Dataset

This project uses the **ECG5000** dataset for anomaly detection.

- The dataset contains **5,000 electrocardiogram samples**
- Each sample has **140 data points**
- Labels are simplified into:
  - `1` = normal rhythm
  - `0` = abnormal rhythm

The objective is to detect abnormal heart rhythms by comparing reconstruction error between normal and anomalous signals.

## Project objective

The main goals of this notebook are to:

1. Build an autoencoder model in TensorFlow/Keras
2. Train the model on normal ECG signals
3. Reconstruct ECG inputs and measure reconstruction loss
4. Use a threshold on reconstruction error to classify anomalies
5. Evaluate performance using metrics such as accuracy, precision, and recall

## Methods

The notebook follows this pipeline:

1. Load and preprocess the ECG5000 dataset
2. Split the data into training, validation, and test sets
3. Train the autoencoder using normal samples only
4. Reconstruct test samples
5. Compute reconstruction loss
6. Classify samples as normal or anomalous based on a threshold
7. Evaluate results with classification metrics and visualizations

## Technologies used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## File structure

```bash
.
├── autoencoders_TF.ipynb
├── ECG5000/
└── README.md
