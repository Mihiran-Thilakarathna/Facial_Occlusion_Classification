import tensorflow as tf
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import json
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

# Load the V2 model
model = tf.keras.models.load_model("models/model_v2.keras")

# Load V2 class names
with open("models/classes_v2.json") as f:
    class_names = json.load(f)

# File picker
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Image files","*.jpg *.png *.jpeg")]
)

if not file_path:
    print("No image selected. Exiting...")
    exit()

print("Selected:", file_path)

# Load image
img = Image.open(file_path).convert("RGB")
img_resized = img.resize((IMG_SIZE, IMG_SIZE))

# PREPROCESSING: The most important difference for V2
img_array = np.array(img_resized)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array) # MobileNetV2 preprocessing (-1 to 1)

# Predict
pred = model.predict(img_array)[0]

index = np.argmax(pred)
confidence = pred[index]
pred_percentages = pred * 100

print(f"\nPrediction: {class_names[index]}")
print(f"Confidence: {confidence*100:.2f} %")

# =====================================
# VISUALIZATION 
# =====================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# 1. Show Uploaded Image
ax1.imshow(img) 
ax1.axis('off')
ax1.set_title('Uploaded Image')

# 2. Show Bar Chart

y_pos = np.arange(len(class_names))

# Using green color for the bars to match the V2 theme
ax2.barh(y_pos, pred_percentages, height=0.5, align='center', color='mediumseagreen')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(class_names)
ax2.invert_yaxis()  
ax2.set_xlabel('Confidence (%)')
ax2.set_xlim(0, 100)
ax2.set_title(f'Predicted: {class_names[index]} ({confidence*100:.1f}%)')
ax2.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()