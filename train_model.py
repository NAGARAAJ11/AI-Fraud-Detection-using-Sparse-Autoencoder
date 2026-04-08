# ==============================
# FRAUD DETECTION PROJECT
# Sparse Autoencoder + Comparison Models
# ==============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# 1. LOAD DATASET
# ==============================

FILE_PATH = "D:\SEM VI\DL\FraudDetectionProject\dataset\creditcard.csv"

if not os.path.exists(FILE_PATH):
    print("Dataset not found!")
    exit()

data = pd.read_csv(FILE_PATH)
data = data.dropna()

print("Dataset shape:", data.shape)

# ==============================
# 2. PREPROCESSING
# ==============================

X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train only on normal transactions
X_train = X_scaled[y == 0]

# Test on full dataset
X_test = X_scaled
y_test = y

# ==============================
# 3. BUILD SPARSE AUTOENCODER
# ==============================

input_dim = X_train.shape[1]

input_layer = layers.Input(shape=(input_dim,))

encoded = layers.Dense(
    20,
    activation="relu",
    activity_regularizer=tf.keras.regularizers.l1(1e-4)
)(input_layer)

encoded = layers.Dense(10, activation="relu")(encoded)

decoded = layers.Dense(20, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = models.Model(
    inputs=input_layer,
    outputs=decoded
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

# ==============================
# 4. TRAIN MODEL
# ==============================

autoencoder.fit(
    X_train,
    X_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    shuffle=True
)

# Save model
autoencoder.save("model.h5")

print("Model saved successfully")

# ==============================
# 5. RECONSTRUCTION ERROR
# ==============================

reconstructions = autoencoder.predict(X_test)

mse = np.mean(
    np.power(X_test - reconstructions, 2),
    axis=1
)

threshold = np.percentile(mse, 95)

y_pred = (mse > threshold).astype(int)

# ==============================
# 6. EVALUATION
# ==============================

print("\n=== Sparse Autoencoder ===")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        mse
    )
)

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==============================
# 7. ISOLATION FOREST
# ==============================

iso = IsolationForest(
    contamination=0.001,
    random_state=42
)

y_pred_iso = iso.fit_predict(X_test)

y_pred_iso = (
    y_pred_iso == -1
).astype(int)

print("\n=== Isolation Forest ===")

print(
    classification_report(
        y_test,
        y_pred_iso
    )
)

# ==============================
# 8. ONE-CLASS SVM
# ==============================

svm = OneClassSVM(
    nu=0.001
)

y_pred_svm = svm.fit_predict(X_test)

y_pred_svm = (
    y_pred_svm == -1
).astype(int)

print("\n=== One-Class SVM ===")

print(
    classification_report(
        y_test,
        y_pred_svm
    )
)

print("\nProject completed successfully")
