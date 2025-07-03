import os
import pandas as pd
import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
image_dir = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\runs\detect\lp_test\crops\license_plate'
output_csv = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\labels_pytesseract.csv'

# Manually set path to tesseract.exe if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Enhanced Preprocessing Function ---
def preprocess_plate(img):
    """Apply advanced preprocessing for license plates"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Moderate upscaling (2x instead of 3x)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Gentle contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    # Light denoising only
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return denoised

# --- OCR Configuration ---
# Try different PSM modes: 7 (single line) or 13 (raw line)
TESS_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# --- Data containers ---
data = []
preview_images = []

# --- Process all images ---
for i, image_filename in enumerate(os.listdir(image_dir)):
    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        failed_count += 1
        continue
            
    img = cv2.imread(image_path)
    if img is None:
        continue
    
    # Apply enhanced preprocessing
    processed_img = preprocess_plate(img)
    
    # Perform OCR with refined configuration
    ocr_text = pytesseract.image_to_string(
        processed_img, 
        config=TESS_CONFIG
    ).strip()
    
    # Post-process OCR results
    ocr_text = ''.join(c for c in ocr_text if c.isalnum())
    
    data.append({'filename': image_filename, 'ocr_text': ocr_text})
    preview_images.append((processed_img, image_filename, ocr_text))

# --- Save results ---
pd.DataFrame(data).to_csv(output_csv, index=False)
print(f"✅ Generated CSV with {len(data)} entries")

# --- Visualization ---
def show_batches(images, batch_size=20):
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        plt.figure(figsize=(20, 20))
        for j, (img, name, text) in enumerate(batch):
            plt.subplot(5, 4, j+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"{name}\n{text}", fontsize=10)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

show_batches(preview_images)
