import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

with open("outputs/history_v1.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy
plt.figure()
plt.plot(history['accuracy'], label="Train")
plt.plot(history['val_accuracy'], label="Validation")
plt.title("Accuracy Curve (V1 - Advanced Custom CNN)")
plt.legend()
plt.show()

# Loss
plt.figure()
plt.plot(history['loss'], label="Train")
plt.plot(history['val_loss'], label="Validation")
plt.title("Loss Curve (V1 - Advanced Custom CNN)")
plt.legend()
plt.show()

# Confusion Matrix
cm = np.load("outputs/confusion_matrix_v1.npy")
with open("models/classes_v1.json") as f:
    class_names = json.load(f)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix (V1)")
plt.show()