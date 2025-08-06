import cv2
import numpy as np
import matplotlib.pyplot as plt import os

# Define functions
def imagenegation(input_image): return 255 - input_image

def imagethresholding(input_image, threshold_value):
_, thresholded_image = cv2.threshold(input_image, threshold_value, 255, cv2.THRESH_BINARY) return thresholded_image
def imagegammacorrection(input_image, gamma): normalized = input_image / 255.0
gamma_corrected = np.power(normalized, gamma) * 255.0 return np.uint8(gamma_corrected)

# Image folder path
image_dir = "C:/Sem-5/DSIP/Exp-1_pic"

# Image data: (filename, threshold, gamma, title prefix) images = [
("ex1_1.png", 100, 0.4, "Image 1"),
("ex1_2.png", 120, 0.6, "Image 2"),
("ex1_3.png", 90, 0.5, "Image 3"),
("ex1_4.png", 80, 0.3, "Image 4"),
("ex1_5.png", 110, 0.7, "Image 5"),
]

# Plotting
for filename, threshold_val, gamma_val, title in images: path = os.path.join(image_dir, filename)
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if image is None:
print(f"Could not load: {filename}") continue

# Process
neg = imagenegation(image)
 
th = imagethresholding(image, threshold_val)
gamma = imagegammacorrection(image, gamma_val)
# Plot all 4 versions plt.figure(figsize=(10, 4)) plt.suptitle(title, fontsize=16)

plt.subplot(1, 4, 1) plt.imshow(image, cmap='gray') plt.title("Original") plt.axis('off')

plt.subplot(1, 4, 2) plt.imshow(neg, cmap='gray') plt.title("Negation") plt.axis('off')

plt.subplot(1, 4, 3) plt.imshow(th, cmap='gray')
plt.title(f"Threshold\n({threshold_val})") plt.axis('off')

plt.subplot(1, 4, 4) plt.imshow(gamma, cmap='gray') plt.title(f"Gamma\n({gamma_val})") plt.axis('off')

plt.tight_layout() plt.show()
