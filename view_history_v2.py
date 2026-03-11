import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Load history from V2
with open("outputs/history_v2.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy Curve
plt.figure()
plt.plot(history['accuracy'], label="Train")
plt.plot(history['val_accuracy'], label="Validation")
plt.title("Accuracy Curve (V2 - MobileNetV2)")
plt.legend()
plt.show()

# Loss Curve
plt.figure()
plt.plot(history['loss'], label="Train")
plt.plot(history['val_loss'], label="Validation")
plt.title("Loss Curve (V2 - MobileNetV2)")
plt.legend()
plt.show()

# Confusion Matrix from V2
cm = np.load("outputs/confusion_matrix_v2.npy")

# Load classes
with open("models/classes_v2.json") as f:
    class_names = json.load(f)

plt.figure(figsize=(6,5))
# Using Greens to match the V2 theme
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Greens")

plt.title("Confusion Matrix (V2 - MobileNetV2)")
plt.show()