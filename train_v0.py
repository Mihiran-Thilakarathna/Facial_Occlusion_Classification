# =====================================
# Version 0: PURE CNN 
# =====================================
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report

# Create folders to keep things organized
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =====================================
# PATHS
# =====================================
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# =====================================
# DATA GENERATORS (No Augmentation)
# =====================================
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

# Using val_gen for clear validation images
val_data = val_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=True 
)

test_data = test_gen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

class_names = list(train_data.class_indices.keys())

with open("models/classes_v0.json", "w") as f:
    json.dump(class_names, f)

print("Classes:", class_names)

# =====================================
# MODEL 
# =====================================
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Layer 1
    Conv2D(8, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),

    # Layer 2
    Conv2D(16, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),

    Flatten(),
    Dense(16, activation='relu'),

    Dense(len(class_names), activation='softmax')
])

# =====================================
# COMPILE
# =====================================
model.compile(
    optimizer=SGD(learning_rate=0.01), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
model.summary()

# =====================================
# TRAIN
# =====================================
history = model.fit(
    train_data, validation_data=val_data, epochs=EPOCHS
)

# =====================================
# SAVE HISTORY & MODEL
# =====================================
with open("outputs/history_v0.pkl", "wb") as f:
    pickle.dump(history.history, f)

model.save("models/model_v0.keras")
print("\nModel and Training history saved successfully!")

# =====================================
# TEST EVALUATION
# =====================================
test_data.reset()
y_prob = model.predict(test_data)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
np.save("outputs/confusion_matrix_v0.npy", cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix (V0: Simplified Pure CNN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix_v0.png")
plt.close()

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n===== Classification Report (V0) =====\n")
print(report)

with open("outputs/classification_report_v0.txt", "w") as f:
    f.write(report)

# =====================================
# VALIDATION IMAGE PREDICTIONS
# =====================================
print("\nShowing validation predictions...")
images, labels = next(val_data)
pred_probs = model.predict(images)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = np.argmax(labels, axis=1)

plt.figure(figsize=(15,8))
for i in range(min(12, len(images))):
    plt.subplot(3,4,i+1)
    plt.imshow(images[i])
    plt.axis('off')

    pred_label = class_names[pred_classes[i]]
    true_label = class_names[true_classes[i]]
    confidence = np.max(pred_probs[i]) * 100
    color = "green" if pred_classes[i] == true_classes[i] else "red"

    plt.title(f"P:{pred_label}\nT:{true_label}\n{confidence:.1f}%", color=color, fontsize=9)

plt.suptitle("Validation Image Predictions (V0: Simplified Pure CNN)")
plt.tight_layout()
plt.savefig("outputs/prediction_grid_v0.png")
plt.close()

# =====================================
# MISCLASSIFIED IMAGES ONLY 
# =====================================
plt.figure(figsize=(15,8))
count = 0

for i in range(len(images)):
    if pred_classes[i] != true_classes[i]:
        plt.subplot(3,4,count+1)
        plt.imshow(images[i])
        plt.axis('off')
        
        plt.title(f"Wrong!\nP:{class_names[pred_classes[i]]}\nT:{class_names[true_classes[i]]}", color="red", fontsize=9)
        
        count += 1
        if count == 12:
            break

if count > 0:
    plt.suptitle("Misclassified Validation Images (V0)")
    plt.tight_layout()
    plt.savefig("outputs/misclassified_v0.png")
plt.close()

print("All V0 outputs including prediction grids are saved successfully!")