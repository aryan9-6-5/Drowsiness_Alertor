import cv2
import matplotlib.pyplot as plt
import os

closed_dir = "D:\projects\PROJECT 1\drowsiness_alertor\data\cew\ClosedEyeimages"
open_dir = "D:\projects\PROJECT 1\drowsiness_alertor\data\cew\OpenEyesimages"

# Pick one image from each category
closed_img_path = os.path.join(closed_dir, os.listdir(closed_dir)[0])
open_img_path = os.path.join(open_dir, os.listdir(open_dir)[0])

# Read using OpenCV (loads in BGR format)
closed_img = cv2.imread(closed_img_path)
open_img = cv2.imread(open_img_path)

# Convert to RGB for matplotlib
closed_img_rgb = cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB)
open_img_rgb = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)

# Plot them
plt.figure(figsize=(10, 4))
plt.suptitle("Sample Images from CEW Dataset", fontsize=16)

plt.subplot(1, 2, 1)
plt.imshow(open_img_rgb)
plt.title("Open Eyes")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(closed_img_rgb)
plt.title("Closed Eyes")
plt.axis('off')

plt.tight_layout()
plt.show()
