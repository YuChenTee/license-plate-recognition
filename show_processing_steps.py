import os

import cv2
import matplotlib.pyplot as plt

# --- Configuration ---
detection_image_path = (
    r"D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\runs\detect\lp_test\cars178.png"
)
license_plate_image_path = r"D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\runs\detect\lp_test\crops\license_plate\Cars178.jpg"
recognized_text = "522 92Z"


# --- Preprocessing Steps for Visualization ---
def preprocess_steps(img):
    steps = {}

    # Step 0: YOLOv5 Detection Result
    if os.path.exists(detection_image_path):
        det_img = cv2.imread(detection_image_path)
        steps["YOLOv5 Detection"] = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)

    # Step 1: Original
    steps["Crop"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 2: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["Grayscale"] = gray

    # Step 3: Resize
    height, width = gray.shape
    scale = max(1, 400 // max(height, width))
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    steps["Resized"] = resized

    # Step 4: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    steps["CLAHE Enhanced"] = enhanced

    # Step 5: Bilateral Filter
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    steps["Bilateral Filtered"] = denoised

    # Step 6: Recognised Text Overlay
    result_img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    annotated = result_img.copy()
    cv2.putText(
        annotated, f"OCR: {recognized_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
    )
    steps["OCR Result"] = annotated

    return steps


# --- Visualization ---
def plot_steps(steps, title="Image"):
    n = len(steps)
    plt.figure()
    for i, (name, img) in enumerate(steps.items()):
        plt.subplot(1, n, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(name, fontsize=10)
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Main ---
img = cv2.imread(license_plate_image_path)
if img is not None:
    print(f"Plotting preprocessing steps for image: {os.path.basename(license_plate_image_path)}")
    steps = preprocess_steps(img)
    plot_steps(steps, title=os.path.basename(license_plate_image_path))
else:
    print(f"Failed to load image: {license_plate_image_path}")
