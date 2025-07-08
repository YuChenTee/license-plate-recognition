import os
import pandas as pd
import easyocr
import cv2
import matplotlib.pyplot as plt

# --- Configuration ---
image_dir = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\runs\detect\lp_test\crops\license_plate'
output_csv = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\labels_easyocr.csv'

reader = easyocr.Reader(['en'], gpu=False)

def preprocess_plate(img):
    """Optimized preprocessing for license plates"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = max(1, 400 // max(height, width))
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

# --- OCR loop ---
data = []
preview_images = []

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

    processed_img = preprocess_plate(img)

    results = reader.readtext(processed_img, detail=0)
    text = results[0] if results else ''

    data.append({'filename': image_filename, 'ocr_text': text})
    preview_images.append((processed_img, image_filename, text))

# --- Save CSV ---
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ EasyOCR finished. Saved to {output_csv}")

# --- Visualization ---
def show_batches(images, batch_size=20):
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        plt.figure()
        for j, (img, name, text) in enumerate(batch):
            plt.subplot(5, 4, j+1)
            plt.imshow(img)
            plt.title(f"{name}\n{text}", fontsize=10)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

show_batches(preview_images, batch_size=20)
