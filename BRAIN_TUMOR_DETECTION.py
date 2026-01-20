#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np 
import cv2
import os
import shutil
import kagglehub
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input as efficientnet_preprocess
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


print("="*70)
print("BRAIN TUMOR DETECTION - VGG16 & EfficientNetB7")
print("="*70)

# DOWNLOADING THE DATASET USING KAGGLEHUB
print("\nDownloading dataset from Kaggle...")


# In[3]:


try:
    dataset_path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
    print(f"Dataset cached at: {dataset_path}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("\nPlease ensure:")
    print("1. You have kaggle.json configured")
    print("2. Internet connection is working")
    dataset_path = "."  # Use current directory as fallback
    print(f"Using fallback path: {dataset_path}")
    raise

# CREATING THE WORKING DIRECTORY
WORK_DIR = 'brain_mri_working'
os.makedirs(WORK_DIR, exist_ok=True)
print(f"\nCreated working directory: {WORK_DIR}")

# FIND AND COPY THE DATASET TO THE WORKING DIRECTORY
print("\nSearching for dataset folders...")
found = False


# In[4]:


for root, dirs, files in os.walk(dataset_path):
    dirs_lower = {d.lower(): d for d in dirs}

    if 'yes' in dirs_lower and 'no' in dirs_lower:
        print(f"Found dataset at: {root}")
        found = True

        for target_name in ['yes', 'no']:
            original_name = dirs_lower[target_name]
            src_path = os.path.join(root, original_name)
            dst_path = os.path.join(WORK_DIR, target_name)

            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)

            os.makedirs(dst_path, exist_ok=True)
            files_copied = 0

            for filename in os.listdir(src_path):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                    src_file = os.path.join(src_path, filename)
                    dst_file = os.path.join(dst_path, filename)
                    try:
                        shutil.copy2(src_file, dst_file)
                        files_copied += 1
                    except Exception as e:
                        print(f"Warning: Could not copy {filename}: {e}")

            print(f"  Copied {target_name}: {files_copied} images")

        break
if not found:
    print("\nERROR: Could not find 'yes' and 'no' folders!")
    raise FileNotFoundError("Dataset folders not found")


# In[5]:


IMG_PATH = WORK_DIR + '/'

# VERIFYING THE DATASET
yes_count = len([f for f in os.listdir(os.path.join(IMG_PATH, 'yes')) if f.endswith(('.jpg', '.jpeg', '.png'))])
no_count = len([f for f in os.listdir(os.path.join(IMG_PATH, 'no')) if f.endswith(('.jpg', '.jpeg', '.png'))])

print(f"\nDataset summary:")
print(f"  YES (tumor): {yes_count} images")
print(f"  NO (no tumor): {no_count} images")
print(f"  Total: {yes_count + no_count} images")
if yes_count == 0 or no_count == 0:
    raise ValueError("Dataset is empty!")

# CONFIGURATION
IMG_SIZE_VGG = (224, 224)
IMG_SIZE_EFFICIENT = (600, 600) 

# EfficientNetB7 uses larger input
RANDOM_SEED = 123
BATCH_SIZE = 32
EPOCHS = 30

# CREATE TRAIN/VAL/TEST DIRECTORIES
print("\nCreating train/val/test directories...")
for split in ['TRAIN', 'VAL', 'TEST']:
    for label in ['YES', 'NO']:
        os.makedirs(f'{split}/{label}', exist_ok=True)


# In[6]:


# THIS PART IS FROM KAGGLEHUB
# SPLIT DATA: 70% TRAIN, 20% VAL, 10% TEST
print("\nSplitting data...")
np.random.seed(RANDOM_SEED)
for class_name in ['yes', 'no']:
    class_path = os.path.join(IMG_PATH, class_name)
    files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    np.random.shuffle(files)

    #TOTAL PERCENTAGE
    n_total = len(files)
    #10% TESTING
    n_test = int(0.1 * n_total)
    #20% VALIDATION
    n_val = int(0.2 * n_total)

    test_files = files[:n_test]
    val_files = files[n_test:n_test + n_val]
    train_files = files[n_test + n_val:]

    for filename in test_files:
        shutil.copy2(os.path.join(class_path, filename), f'TEST/{class_name.upper()}/{filename}')

    for filename in val_files:
        shutil.copy2(os.path.join(class_path, filename), f'VAL/{class_name.upper()}/{filename}')

    for filename in train_files:
        shutil.copy2(os.path.join(class_path, filename), f'TRAIN/{class_name.upper()}/{filename}')

    print(f"  {class_name.upper()}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("Data split complete!")


# In[7]:


# THIS PART IS FROM KAGGLE
# LOADING DATA FUNCTION
def load_data(directory, img_size):
    images = []
    labels = []
    label_dict = {'NO': 0, 'YES': 1}

    for label_name in os.listdir(directory):
        label_path = os.path.join(directory, label_name)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
                        images.append(img)
                        labels.append(label_dict[label_name])

    return np.array(images), np.array(labels)


# In[8]:


# LOADING DATASETS FOR BOTH MODELS
print("\nLoading datasets for VGG16 (224x224)...")
X_train_vgg, y_train = load_data('TRAIN/', IMG_SIZE_VGG)
X_val_vgg, y_val = load_data('VAL/', IMG_SIZE_VGG)
X_test_vgg, y_test = load_data('TEST/', IMG_SIZE_VGG)

print(f"Train: {X_train_vgg.shape[0]} images")
print(f"Val: {X_val_vgg.shape[0]} images")
print(f"Test: {X_test_vgg.shape[0]} images")

print("\nLoading datasets for EfficientNetB7 (600x600)...")
X_train_eff, _ = load_data('TRAIN/', IMG_SIZE_EFFICIENT)
X_val_eff, _ = load_data('VAL/', IMG_SIZE_EFFICIENT)
X_test_eff, _ = load_data('TEST/', IMG_SIZE_EFFICIENT)


# In[9]:


# VISUALISING DISTRIBUTION
fig, ax = plt.subplots(figsize=(10, 6))
sets = ['Train', 'Validation', 'Test']
no_counts = [np.sum(y_train == 0), np.sum(y_val == 0), np.sum(y_test == 0)]
yes_counts = [np.sum(y_train == 1), np.sum(y_val == 1), np.sum(y_test == 1)]

x_pos = np.arange(len(sets))
width = 0.35

ax.bar(x_pos - width/2, no_counts, width, label='No Tumor', color='#33cc33', alpha=0.8)
ax.bar(x_pos + width/2, yes_counts, width, label='Tumor', color='#ff3300', alpha=0.8)

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(sets)
ax.legend()
plt.tight_layout()
plt.show()


# In[10]:


#THIS PART IS FROM KAGGLE
# CROPPING AND PREPROCESSING FUNCTION
def crop_and_preprocess(image, img_size, preprocess_func):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = image[y:y+h, x:x+w]
    else:
        cropped = image

    resized = cv2.resize(cropped, img_size, interpolation=cv2.INTER_CUBIC)
    return preprocess_func(resized)
    print("\nPreprocessing images for VGG16...")


# In[11]:


X_train_vgg_prep = np.array([crop_and_preprocess(img, IMG_SIZE_VGG, vgg_preprocess) for img in X_train_vgg])
X_val_vgg_prep = np.array([crop_and_preprocess(img, IMG_SIZE_VGG, vgg_preprocess) for img in X_val_vgg])
X_test_vgg_prep = np.array([crop_and_preprocess(img, IMG_SIZE_VGG, vgg_preprocess) for img in X_test_vgg])
print("Preprocessing images for EfficientNetB7...")
X_train_eff_prep = np.array([crop_and_preprocess(img, IMG_SIZE_EFFICIENT, efficientnet_preprocess) for img in X_train_eff])
X_val_eff_prep = np.array([crop_and_preprocess(img, IMG_SIZE_EFFICIENT, efficientnet_preprocess) for img in X_val_eff])
X_test_eff_prep = np.array([crop_and_preprocess(img, IMG_SIZE_EFFICIENT, efficientnet_preprocess) for img in X_test_eff])
print("Done!")


# In[12]:


# ============================================================================
# MODEL 1: VGG16 (FOR GUI)
# ============================================================================
print("\n" + "="*70)
print("TRAINING VGG16 MODEL (FOR GUI)")
print("="*70)

base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE_VGG + (3,))
base_model_vgg.trainable = False

model_vgg = Sequential([
    base_model_vgg,
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)
print("\nVGG16 Model Summary:")
model_vgg.summary()
train_datagen_vgg = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)


# In[13]:


train_gen_vgg = train_datagen_vgg.flow(X_train_vgg_prep, y_train, batch_size=BATCH_SIZE)
val_gen_vgg = ImageDataGenerator().flow(X_val_vgg_prep, y_val, batch_size=16, shuffle=False)

early_stop_vgg = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)

print("\nTraining VGG16...")
history_vgg = model_vgg.fit(
    train_gen_vgg,
    steps_per_epoch=len(X_train_vgg_prep) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen_vgg,
    validation_steps=len(X_val_vgg_prep) // 16,
    callbacks=[early_stop_vgg],
    verbose=1
)


# In[14]:


# EVALUATING VGG16
test_pred_vgg = (model_vgg.predict(X_test_vgg_prep, verbose=0) > 0.5).astype(int).flatten()
test_acc_vgg = accuracy_score(y_test, test_pred_vgg)
print(f"\nVGG16 Test Accuracy: {test_acc_vgg*100:.2f}%")# SAVING VGG16 MODEL FOR GUI
model_vgg.save('brain_tumor_vgg16_model.h5')
print("✓ VGG16 Model saved: brain_tumor_vgg16_model.h5 (FOR GUI)")


# In[15]:


# ============================================================================
# VISUALIZATIONS FOR VGG16
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS FOR VGG16")
print("="*70)


# In[16]:


# PLOTTING TRAINING HISTORY VGG16
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_vgg = range(1, len(history_vgg.history['accuracy']) + 1)

ax1.plot(epochs_vgg, history_vgg.history['accuracy'], 'b-', label='Train', linewidth=2)
ax1.plot(epochs_vgg, history_vgg.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
ax1.set_title('VGG16 - Accuracy', fontweight='bold', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs_vgg, history_vgg.history['loss'], 'b-', label='Train', linewidth=2)
ax2.plot(epochs_vgg, history_vgg.history['val_loss'], 'r-', label='Validation', linewidth=2)
ax2.set_title('VGG16 - Loss', fontweight='bold', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[17]:


# CONFUSION MATRIX FOR VGG16
cm_vgg = confusion_matrix(y_test, test_pred_vgg)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('VGG16 - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[18]:


# HISTOGRAM OF PREDICTIONS FOR VGG16
print("\nGenerating VGG16 sample predictions...")
pred_probs_vgg = model_vgg.predict(X_test_vgg_prep, verbose=0).flatten()
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pred_probs_vgg[y_test == 0], bins=30, alpha=0.6, label='No Tumor', color='green')
ax.hist(pred_probs_vgg[y_test == 1], bins=30, alpha=0.6, label='Tumor', color='red')
ax.set_xlabel('Prediction Confidence', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('VGG16 - Prediction Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[19]:


# THIS PART IS FROM KAGGLE
# Create indices for VGG16 samples
np.random.seed(42)
n_samples = 8
indices_vgg = np.random.choice(len(X_test_vgg), n_samples, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    i = indices_vgg[idx]
    img = X_test_vgg[i]  # Use original images, not preprocessed
    true_label = y_test[i]
    pred_prob = pred_probs_vgg[i]
    pred_label = 1 if pred_prob > 0.5 else 0

    ax.imshow(img)
    ax.axis('off')

    true_text = "Tumor" if true_label == 1 else "No Tumor"
    pred_text = "Tumor" if pred_label == 1 else "No Tumor"
    confidence = pred_prob if pred_label == 1 else (1 - pred_prob)

    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'True: {true_text}\nPred: {pred_text} ({confidence*100:.1f}%)',
                 fontsize=10, color=color, fontweight='bold')

plt.suptitle('VGG16 - Sample Predictions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[20]:


# ============================================================================
# MODEL 2: EfficientNetB7 (FOR COMPARISON & VISUALIZATION)
# ============================================================================
print("\n" + "="*70)
print("TRAINING EfficientNetB7 MODEL (FOR COMPARISON)")
print("="*70)
base_model_eff = EfficientNetB7(weights='imagenet', include_top=False, input_shape=IMG_SIZE_EFFICIENT + (3,))
base_model_eff.trainable = False

model_eff = Sequential([
    base_model_eff,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_eff.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy'])
print("\nEfficientNetB7 Model Summary:")
model_eff.summary()
train_datagen_eff = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)


# In[21]:


train_gen_eff = train_datagen_eff.flow(X_train_eff_prep, y_train, batch_size=8)  # Smaller batch for larger images
val_gen_eff = ImageDataGenerator().flow(X_val_eff_prep, y_val, batch_size=8, shuffle=False)

early_stop_eff = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)

print("\nTraining EfficientNetB7...")
history_eff = model_eff.fit(
    train_gen_eff,
    steps_per_epoch=len(X_train_eff_prep) // 8,
    epochs=EPOCHS,
    validation_data=val_gen_eff,
    validation_steps=len(X_val_eff_prep) // 8,
    callbacks=[early_stop_eff],
    verbose=1
)


# In[22]:


# EVALUATING EfficientNetB7
test_pred_eff = (model_eff.predict(X_test_eff_prep, verbose=0) > 0.5).astype(int).flatten()
test_acc_eff = accuracy_score(y_test, test_pred_eff)
print(f"\nEfficientNetB7 Test Accuracy: {test_acc_eff*100:.2f}%")


# In[23]:


# ============================================================================
# VISUALIZATIONS FOR EfficientNetB7
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS FOR EfficientNetB7")
print("="*70)
# PLOTTING TRAINING HISTORY EfficientNetB7
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_eff = range(1, len(history_eff.history['accuracy']) + 1)

ax1.plot(epochs_eff, history_eff.history['accuracy'], 'b-', label='Train', linewidth=2)
ax1.plot(epochs_eff, history_eff.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
ax1.set_title('EfficientNetB7 - Accuracy', fontweight='bold', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs_eff, history_eff.history['loss'], 'b-', label='Train', linewidth=2)
ax2.plot(epochs_eff, history_eff.history['val_loss'], 'r-', label='Validation', linewidth=2)
ax2.set_title('EfficientNetB7 - Loss', fontweight='bold', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[24]:


# CONFUSION MATRIX FOR EfficientNetB7
cm_eff = confusion_matrix(y_test, test_pred_eff)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_eff, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('EfficientNetB7 - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[25]:


# HISTOGRAM OF PREDICTIONS FOR EfficientNetB7
pred_probs_eff = model_eff.predict(X_test_eff_prep, verbose=0).flatten()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pred_probs_eff[y_test == 0], bins=30, alpha=0.6, label='No Tumor', color='green')
ax.hist(pred_probs_eff[y_test == 1], bins=30, alpha=0.6, label='Tumor', color='red')
ax.set_xlabel('Prediction Confidence', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('EfficientNetB7 - Prediction Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[26]:


# ============================================================================
# SAMPLE PREDICTIONS VISUALIZATION
# ============================================================================
print("\nGenerating sample predictions visualization...")

# Select random samples
np.random.seed(42)
n_samples = 8
indices = np.random.choice(len(X_test_eff), n_samples, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    i = indices[idx]
    img = X_test_eff[i]
    true_label = y_test[i]
    pred_prob = pred_probs_eff[i]
    pred_label = 1 if pred_prob > 0.5 else 0

    ax.imshow(img)
    ax.axis('off')

    true_text = "Tumor" if true_label == 1 else "No Tumor"
    pred_text = "Tumor" if pred_label == 1 else "No Tumor"
    confidence = pred_prob if pred_label == 1 else (1 - pred_prob)

    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'True: {true_text}\nPred: {pred_text} ({confidence*100:.1f}%)',
                 fontsize=10, color=color, fontweight='bold')

plt.suptitle('EfficientNetB7 - Sample Predictions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[27]:


# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison_data = {
    'Model': ['VGG16 (GUI)', 'EfficientNetB7'],
    'Test Accuracy': [f'{test_acc_vgg*100:.2f}%', f'{test_acc_eff*100:.2f}%'],
    'Input Size': ['224x224', '600x600'],
    'Parameters': [f'{model_vgg.count_params():,}', f'{model_eff.count_params():,}']
}

print("\n{:<20} {:<20} {:<15} {:<20}".format('Model', 'Test Accuracy', 'Input Size', 'Parameters'))
print("-" * 75)
for i in range(len(comparison_data['Model'])):
    print("{:<20} {:<20} {:<15} {:<20}".format(
        comparison_data['Model'][i],
        comparison_data['Test Accuracy'][i],
        comparison_data['Input Size'][i],
        comparison_data['Parameters'][i]
    ))


# In[28]:


# BAR CHART COMPARISON
fig, ax = plt.subplots(figsize=(10, 6))
models = ['VGG16\n(GUI)', 'EfficientNetB7']
accuracies = [test_acc_vgg * 100, test_acc_eff * 100]
colors = ['#3498db', '#e74c3c']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


# In[29]:


# CLEANING UP
print("\n" + "="*70)
print("CLEANUP")
print("="*70)
for folder in ['TRAIN', 'VAL', 'TEST', WORK_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
print("✓ Temporary folders removed")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"VGG16 (FOR GUI):        {test_acc_vgg*100:.2f}% accuracy")
print(f"EfficientNetB7:         {test_acc_eff*100:.2f}% accuracy")
print("\nFiles saved:")
print("  - brain_tumor_vgg16_model.h5 (Use this for GUI)")
print("="*70)


# In[ ]:


import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from datetime import datetime
import os

class BrainTumorDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection & Advisory System")
        self.root.geometry("1100x850")
        self.root.configure(bg='#f0f0f0')

        # --- CONFIGURATION ---
        self.MODEL_PATH = 'brain_tumor_vgg16_model.h5'
        self.IMG_SIZE = (224, 224)

        self.model = None
        self.load_model_file()
        self.setup_ui()

        self.current_image_tk = None
        self.original_cv_image = None

    def load_model_file(self):
        try:
            if os.path.exists(self.MODEL_PATH):
                self.model = load_model(self.MODEL_PATH)
                print("✓ Model loaded successfully!")
            else:
                print(f"✗ Model file {self.MODEL_PATH} not found.")
        except Exception as e:
            messagebox.showerror("Model Error", f"Error loading model: {e}")

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame, text="🧠 Brain Tumor Detection & Advisory System",
                               font=("Arial", 22, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)

        # Main content
        content_frame = tk.Frame(self.root, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Panel (Image)
        left_frame = tk.Frame(content_frame, bg='white', relief=tk.RIDGE, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(left_frame, text="MRI Image Preview", font=("Arial", 14, "bold"), bg='white', fg='#2c3e50').pack(pady=10)

        self.image_canvas = tk.Canvas(left_frame, width=450, height=450, bg='#ecf0f1', highlightthickness=0)
        self.image_canvas.pack(pady=10)
        self.image_canvas.create_text(225, 225, text="Step 1: Upload MRI\nStep 2: Click Analyze", 
                                     font=("Arial", 12), fill='#95a5a6', justify=tk.CENTER, tags="placeholder")

        # Buttons Container
        btn_frame = tk.Frame(left_frame, bg='white')
        btn_frame.pack(fill=tk.X, padx=20, pady=10)

        self.upload_btn = tk.Button(btn_frame, text="📁 Upload MRI Image", font=("Arial", 12, "bold"), 
                                    bg='#3498db', fg='white', cursor='hand2', command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.analyze_btn = tk.Button(btn_frame, text="⚡ Analyze Image", font=("Arial", 12, "bold"), 
                                     bg='#e67e22', fg='white', cursor='hand2', state=tk.DISABLED, command=self.run_detection)
        self.analyze_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Right Panel (Results)
        right_frame = tk.Frame(content_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        results_frame = tk.Frame(right_frame, bg='white', relief=tk.RIDGE, bd=2)
        results_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(results_frame, text="AI Diagnostic Output", font=("Arial", 15, "bold"), bg='white', fg='#2c3e50').pack(pady=10)

        self.prediction_label = tk.Label(results_frame, text="Awaiting Analysis...", font=("Arial", 18, "bold"), bg='white', fg='#7f8c8d')
        self.prediction_label.pack(pady=5)

        self.confidence_text = tk.Label(results_frame, text="---%", font=("Arial", 24, "bold"), bg='white', fg='#2c3e50')
        self.confidence_text.pack()

        self.progress_canvas = tk.Canvas(results_frame, width=280, height=20, bg='#ecf0f1', highlightthickness=0)
        self.progress_canvas.pack(pady=10)

        self.timestamp_label = tk.Label(results_frame, text="", font=("Arial", 9), bg='white', fg='#7f8c8d')
        self.timestamp_label.pack(pady=5)

        # Recommendations
        rec_frame = tk.Frame(right_frame, bg='white', relief=tk.RIDGE, bd=2)
        rec_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(rec_frame, text="Full Medical Advisory", font=("Arial", 14, "bold"), bg='white').pack(pady=10)

        self.recommendations_text = scrolledtext.ScrolledText(rec_frame, wrap=tk.WORD, font=("Arial", 10), bg='#f8f9fa', state=tk.DISABLED)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        tk.Button(right_frame, text="🗑️ Reset System", bg='#95a5a6', fg='white', command=self.clear_results).pack(pady=10, fill=tk.X)

    def crop_and_preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped = image[y:y+h, x:x+w]
        else:
            cropped = image

        resized = cv2.resize(cropped, self.IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        img_array = np.expand_dims(resized, axis=0)
        return vgg_preprocess(img_array.astype('float32'))

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path: return

        try:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_cv_image = img
            self.display_image(img)

            # Enable the analyze button now that we have an image
            self.analyze_btn.config(state=tk.NORMAL, bg='#27ae60')
            self.prediction_label.config(text="Ready to Analyze", fg='#2980b9')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def run_detection(self):
        if self.original_cv_image is None: return

        self.prediction_label.config(text="🔍 Analyzing...", fg='#f39c12')
        self.root.update_idletasks() # Force UI update

        try:
            prep_img = self.crop_and_preprocess(self.original_cv_image)
            prediction_score = self.model.predict(prep_img, verbose=0)[0][0]
            self.update_results(prediction_score)
        except Exception as e:
            messagebox.showerror("Processing Error", f"Model failed to predict: {e}")

    def update_results(self, score):
        has_tumor = score > 0.5
        confidence = score if has_tumor else (1 - score)

        res_text = "⚠️ TUMOR DETECTED" if has_tumor else "✅ NO TUMOR DETECTED"
        res_color = "#e74c3c" if has_tumor else "#27ae60"

        self.prediction_label.config(text=res_text, fg=res_color)
        self.confidence_text.config(text=f"{confidence*100:.2f}%", fg=res_color)

        # Progress bar
        self.progress_canvas.delete("bar")
        self.progress_canvas.create_rectangle(0, 0, int(280 * confidence), 20, fill=res_color, tags="bar")

        self.timestamp_label.config(text=f"Last Analysis: {datetime.now().strftime('%H:%M:%S | %d %b %Y')}")

        # Recommendations
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, self.generate_detailed_advisory(has_tumor, confidence))
        self.recommendations_text.config(state=tk.DISABLED)

    def display_image(self, img):
        h, w = img.shape[:2]
        ratio = min(450/w, 450/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img_pil = Image.fromarray(cv2.resize(img, (new_w, new_h)))
        self.current_image_tk = ImageTk.PhotoImage(img_pil)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(225, 225, image=self.current_image_tk, anchor=tk.CENTER)

    def generate_detailed_advisory(self, has_tumor, conf):
        header = "DETAILED MEDICAL ADVISORY REPORT\n" + "="*40 + "\n\n"

        if has_tumor:
            content = (
                f"STATUS: High Probability of Mass Detection ({conf*100:.1f}%)\n\n"
                "1. IMMEDIATE CLINICAL ACTION:\n"
                "   • Schedule an appointment with a Neuro-Oncologist or Neurosurgeon.\n"
                "   • Do not interpret this as a final diagnosis; AI can have false positives.\n\n"
                "2. SUGGESTED FURTHER DIAGNOSTICS:\n"
                "   • Contrast-Enhanced MRI (Gadolinium) for clearer border definition.\n"
                "   • MR Spectroscopy to analyze chemical metabolism of the tissue.\n"
                "   • PET Scan to check for metabolic activity (grading).\n\n"
                "3. SYMPTOMS TO MONITOR:\n"
                "   • Persistent morning headaches or nausea.\n"
                "   • New onset of seizures or localized weakness (limbs).\n"
                "   • Cognitive changes or sudden personality shifts.\n\n"
                "4. LIFESTYLE ADVICE:\n"
                "   • Maintain a 'Brain-Healthy' diet (low inflammation, high antioxidants).\n"
                "   • Ensure high-quality sleep (7-9 hours) to manage intracranial pressure.\n\n"
                "DISCLAIMER: This system is a Screening Tool. A physician MUST confirm results."
            )
        else:
            content = (
                f"STATUS: No Mass Detected ({conf*100:.1f}%)\n\n"
                "1. CLINICAL FOLLOW-UP:\n"
                "   • If you are experiencing symptoms (headaches, vision issues), please consult a physician regardless of this result.\n"
                "   • AI may miss very small or low-grade abnormalities.\n\n"
                "2. PREVENTATIVE BRAIN HEALTH:\n"
                "   • Diet: Focus on Omega-3 fatty acids (Walnuts, Salmon, Flaxseeds).\n"
                "   • Cognitive Exercise: Engage in puzzles, reading, or learning new skills.\n"
                "   • Physical Exercise: Cardiovascular health is directly linked to brain oxygenation.\n\n"
                "3. RISK REDUCTION:\n"
                "   • Minimize exposure to unnecessary radiation.\n"
                "   • Ensure blood pressure is managed, as hypertension affects brain tissue health.\n\n"
                "4. NEXT STEPS:\n"
                "   • File this report. If new symptoms appear in the future, perform a re-scan.\n\n"
                "DISCLAIMER: For screening purposes only. Not a substitute for professional medical advice."
            )
        return header + content

    def clear_results(self):
        self.image_canvas.delete("all")
        self.image_canvas.create_text(225, 225, text="No image loaded", fill='#95a5a6')
        self.prediction_label.config(text="Awaiting Analysis...", fg='#7f8c8d')
        self.confidence_text.config(text="---%", fg='#2c3e50')
        self.progress_canvas.delete("bar")
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED, bg='#95a5a6')
        self.original_cv_image = None

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorDetectorGUI(root)
    root.mainloop()


# In[ ]:




