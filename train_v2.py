# =====================================
# Version 2 - TRANSFER LEARNING (MOBILENETV2)
# (WITH DATA AUGMENTATION, PRE-TRAINED WEIGHTS & EARLY STOPPING)
# =====================================

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20 # 20 is enough for Transfer Learning

# =====================================
# DATA GENERATORS (With Augmentation & MobileNet Preprocessing)
# =====================================
# MobileNetV2 uses 'preprocess_input' instead of 'rescale=1./255'
# Adding Augmentation only for training
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Validation generator MUST NOT have augmentation, only preprocessing
val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    validation_split=0.2
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

val_data = val_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=True 
)

test_data = test_gen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

class_names = list(train_data.class_indices.keys())
with open("models/classes_v2.json", "w") as f:
    json.dump(class_names, f)

# =====================================
# MODEL (MobileNetV2 Base + Custom Head)
# =====================================
# 1. Load the pre-trained MobileNetV2 model (without the final 1000-class layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# 2. Freeze the base model so its learned weights are not updated
base_model.trainable = False

# 3. Create our custom output layers
x = base_model.output
x = GlobalAveragePooling2D()(x) # Better alternative to Flatten for MobileNet
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(len(class_names), activation='softmax')(x)

# 4. Combine base and top to create the final model
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# =====================================
# TRAIN
# =====================================
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stop, reduce_lr])

with open("outputs/history_v2.pkl", "wb") as f:
    pickle.dump(history.history, f)

model.save("models/model_v2.keras")

# =====================================
# TEST EVALUATION
# =====================================
test_data.reset()
y_prob = model.predict(test_data)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
np.save("outputs/confusion_matrix_v2.npy", cm)

plt.figure(figsize=(6,5))
# Changed to 'Greens' so it's easy to distinguish V2 from V1 in the report
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Greens")
plt.title("Confusion Matrix (V2 - MobileNetV2)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix_v2.png")
plt.close()

report = classification_report(y_true, y_pred, target_names=class_names)
with open("outputs/classification_report_v2.txt", "w") as f:
    f.write(report)

# =====================================
# VALIDATION IMAGE PREDICTIONS
# =====================================
images, labels = next(val_data)
pred_probs = model.predict(images)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = np.argmax(labels, axis=1)

plt.figure(figsize=(15,8))
for i in range(min(12, len(images))):
    plt.subplot(3,4,i+1)
    # Undo preprocess_input mapping (-1 to 1) back to (0 to 1) for visualization
    img_viz = (images[i] + 1) / 2.0 
    img_viz = np.clip(img_viz, 0, 1)

    plt.imshow(img_viz)
    plt.axis('off')
    pred_label = class_names[pred_classes[i]]
    true_label = class_names[true_classes[i]]
    confidence = np.max(pred_probs[i]) * 100
    color = "green" if pred_classes[i] == true_classes[i] else "red"
    plt.title(f"P:{pred_label}\nT:{true_label}\n{confidence:.1f}%", color=color, fontsize=9)

plt.suptitle("Validation Image Predictions (V2)")
plt.tight_layout()
plt.savefig("outputs/prediction_grid_v2.png")
plt.close()

# =====================================
# MISCLASSIFIED IMAGES ONLY 
# =====================================
plt.figure(figsize=(15,8))
count = 0

for i in range(len(images)):
    if pred_classes[i] != true_classes[i]:
        plt.subplot(3,4,count+1)
        img_viz = (images[i] + 1) / 2.0 
        img_viz = np.clip(img_viz, 0, 1)
        
        plt.imshow(img_viz)
        plt.axis('off')
        plt.title(f"Wrong!\nP:{class_names[pred_classes[i]]}\nT:{class_names[true_classes[i]]}", color="red", fontsize=9)
        count += 1
        if count == 12:
            break

if count > 0:
    plt.suptitle("Misclassified Validation Images (V2)")
    plt.tight_layout()
    plt.savefig("outputs/misclassified_v2.png")

plt.close()

print("All V2 outputs including prediction grids are saved successfully!")