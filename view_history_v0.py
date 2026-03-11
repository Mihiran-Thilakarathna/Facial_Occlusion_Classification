import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Load history from the outputs folder
with open("outputs/history_v0.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy
plt.figure()
plt.plot(history['accuracy'], label="Train")
plt.plot(history['val_accuracy'], label="Validation")
plt.title("Accuracy Curve (V0 - Pure CNN)")
plt.legend()
plt.show()

# Loss
plt.figure()
plt.plot(history['loss'], label="Train")
plt.plot(history['val_loss'], label="Validation")
plt.title("Loss Curve (V0 - Pure CNN)")
plt.legend()
plt.show()

# Confusion Matrix from the outputs folder
cm = np.load("outputs/confusion_matrix_v0.npy")

# Load classes from the models folder
with open("models/classes_v0.json") as f:
    class_names = json.load(f)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.title("Confusion Matrix (V0)")
plt.show()