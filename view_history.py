import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Load history
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy
plt.figure()
plt.plot(history['accuracy'], label="Train")
plt.plot(history['val_accuracy'], label="Validation")
plt.title("Accuracy Curve")
plt.legend()
plt.show()

# Loss
plt.figure()
plt.plot(history['loss'], label="Train")
plt.plot(history['val_loss'], label="Validation")
plt.title("Loss Curve")
plt.legend()
plt.show()

# Confusion Matrix
cm = np.load("confusion_matrix.npy")

with open("classes.json") as f:
    class_names = json.load(f)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.title("Confusion Matrix")
plt.show()