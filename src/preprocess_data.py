import cv2
import os
import numpy as np

# Path to folders
closed_dir = "D:\projects\PROJECT 1\drowsiness_alertor\data\cew\ClosedEyeimages"
open_dir = "D:\projects\PROJECT 1\drowsiness_alertor\data\cew\OpenEyesimages"

# Initialize lists
images = []
labels = []

def preprocess_folder(folder_path, label):
    print(f"Loading images from {folder_path} with label {label}...")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Failed to read {img_path}")
            continue

        # Resize to 224x224
        img = cv2.resize(img, (224, 224))

        # Normalize to 0-1 range
        img = img / 255.0

        images.append(img)
        labels.append(label)

        # Optional: print status every 100 images
        if len(images) % 100 == 0:
            print(f"{len(images)} images processed...")

# Run for both folders
preprocess_folder(open_dir, label=0)
preprocess_folder(closed_dir, label=1)

# Convert to NumPy arrays
X = np.array(images).reshape(-1, 224, 224, 1)  # 1-channel (grayscale)
y = np.array(labels)

print("Data shape:", X.shape)
print("Label shape:", y.shape)
print("Sample labels:", y[:10])

# Correct path: go one level up to access the real data/ folder
output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)



print("✅ Preprocessing complete. Data saved to 'data/' folder.")
