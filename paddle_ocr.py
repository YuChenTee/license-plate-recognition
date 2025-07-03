import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# --- Configuration ---
image_dir = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\runs\detect\lp_test\crops\license_plate'
output_csv = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\labels_paddleocr.csv'

# Initialize PaddleOCR
print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

# --- Preprocessing ---
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

# --- PaddleOCR Runner ---
def run_paddle_ocr(img):
    """Run PaddleOCR with confidence extraction"""
    try:
        img_array = np.array(img) if not isinstance(img, np.ndarray) else img
        results = paddle_ocr.predict(img_array)
        
        if not results or len(results) == 0:
            return "", 0.0
        
        extracted_texts = []
        total_confidence = 0
        valid_detections = 0
        
        for result in results:
            if 'rec_texts' in result and 'rec_scores' in result:
                for text, score in zip(result['rec_texts'], result['rec_scores']):
                    if score > 0.5:
                        extracted_texts.append(text.upper())
                        total_confidence += score
                        valid_detections += 1
                        print(f"    PaddleOCR detected: '{text}' (confidence: {score:.2f})")
        
        if valid_detections == 0:
            return "", 0.0
            
        combined_text = "".join(extracted_texts)
        avg_confidence = total_confidence / valid_detections
        return combined_text, avg_confidence
        
    except Exception as e:
        print(f"PaddleOCR failed: {e}")
        return "", 0.0

# --- Main OCR Handler ---
def perform_ocr(image_path):
    print(f"Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Failed to load image: {image_path}")
        return ""

    processed_img = preprocess_plate(img)
    text, conf = run_paddle_ocr(processed_img)
    print(f"  Final result: {text}")
    return text

# --- Batch Processing ---
data = []
preview_images = []
failed_count = 0

print("Starting OCR processing using PaddleOCR only...")

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for i, image_filename in enumerate(image_files):
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        failed_count += 1
        continue

    print(f"\n=== Processing {i+1}: {image_filename} ===")
    ocr_text = perform_ocr(image_path)

    data.append({
        'filename': image_filename,
        'ocr_text': ocr_text,
        'text_length': len(ocr_text)
    })

    img = cv2.imread(image_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preview_images.append((img_rgb, image_filename, ocr_text))

# --- Save CSV ---
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"\n✅ PaddleOCR-only run completed. Results saved to {output_csv}")
print(f"Processed {len(data)} plates, {failed_count} files missing")

# --- Statistics ---
print("\n--- OCR Statistics ---")
print(f"Average text length: {df['text_length'].mean():.1f}")
print(f"Empty results: {len(df[df['text_length'] == 0])}")
print("\nResults:")
for idx, row in df.iterrows():
    print(f"  {row['filename']}: '{row['ocr_text']}'")

# --- Visual Diagnostics ---
def plot_results(images, batch_size=20):
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        n = len(batch)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(15, 5 * rows))

        for j, (img, name, text) in enumerate(batch):
            plt.subplot(rows, cols, j + 1)
            plt.imshow(img)
            plt.title(f"{name}\nOCR: '{text}'", fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if preview_images:
    plot_results(preview_images)
