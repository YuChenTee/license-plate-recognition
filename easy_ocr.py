import os
import pandas as pd
import easyocr
import cv2
import matplotlib.pyplot as plt

# --- Configuration ---
image_dir = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\runs\detect\lp_test\crops\license_plate'
output_csv = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\labels_easyocr.csv'

reader = easyocr.Reader(['en'], gpu=False)

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

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = reader.readtext(img_rgb, detail=0)
    text = results[0] if results else ''

    data.append({'filename': image_filename, 'ocr_text': text})
    preview_images.append((img_rgb, image_filename, text))

# --- Save CSV ---
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ EasyOCR finished. Saved to {output_csv}")

# --- Show images (optional) ---
def show_batches(images, batch_size=20):
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        n = len(batch)
        rows = (n + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        axes = axes.flatten()
        for j in range(n):
            img, name, text = batch[j]
            axes[j].imshow(img)
            axes[j].set_title(f"{name}\nOCR: {text}", fontsize=8)
            axes[j].axis('off')
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

show_batches(preview_images, batch_size=20)
