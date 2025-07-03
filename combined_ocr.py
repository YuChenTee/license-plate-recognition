import os
import pandas as pd
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import pytesseract
import re

# --- Configuration ---
image_dir = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\runs\detect\lp_test\crops\license_plate'
output_csv = r'D:\Lecture notes and exercises\Computer Vision\license-plate-recognition\yolov5\labels_easyocr_improved.csv'

# Initialize OCR engines
print("Initializing OCR engines...")
easy_reader = easyocr.Reader(['en'], model_storage_directory='model', download_enabled=True)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Set tesseract path if needed (uncomment and modify path if required)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Enhanced Preprocessing ---
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

def preprocess_for_tesseract(img):
    """Specialized preprocessing for Tesseract"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to improve OCR accuracy
    height, width = gray.shape
    scale = max(2, 800 // max(height, width))
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Apply threshold for better text extraction
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

# --- OCR Functions with Confidence ---
def run_easyocr(img):
    """Run EasyOCR with confidence extraction"""
    try:
        # Get detailed results including confidence
        detailed_results = easy_reader.readtext(img, detail=1, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Combine all detected texts with their confidences
        combined_text = ""
        total_confidence = 0
        valid_detections = 0
        
        for detection in detailed_results:
            text = detection[1].upper().strip()
            confidence = detection[2]
            
            if confidence > 0.5:  # Only consider confident detections
                combined_text += text
                total_confidence += confidence
                valid_detections += 1
                print(f"    EasyOCR detected: '{text}' (confidence: {confidence:.2f})")
        
        if valid_detections == 0:
            return "", 0.0
        
        avg_confidence = total_confidence / valid_detections
        return combined_text, avg_confidence
        
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        return "", 0.0

def run_paddle_ocr(img):
    """Run PaddleOCR with confidence extraction and proper text ordering"""
    try:
        img_array = np.array(img) if not isinstance(img, np.ndarray) else img
        results = paddle_ocr.predict(img_array)
        
        if not results or len(results) == 0:
            return "", 0.0
        
        # Collect all detections with their positions and scores
        detections = []
        for result in results:
            if not all(key in result for key in ['rec_texts', 'rec_scores', 'rec_boxes']):
                continue
                
            for i in range(len(result['rec_texts'])):
                try:
                    text = result['rec_texts'][i]
                    score = result['rec_scores'][i]
                    box = result['rec_boxes'][i]
                    
                    if score <= 0.5:
                        continue
                        
                    # Handle the [x1,y1,x2,y2] format
                    if isinstance(box, np.ndarray) and box.size == 4:
                        x1, y1, x2, y2 = box
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                    else:
                        print(f"Unexpected box format: {box}")
                        continue
                    
                    detections.append({
                        'text': text.upper(),
                        'score': score,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
                    print(f"    PaddleOCR detected: '{text}' (score: {score:.2f}) at ({x_center:.1f},{y_center:.1f}) size: {width}x{height}")
                
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue
        
        if not detections:
            return "", 0.0
            
        # Improved sorting that groups by lines first
        # Sort by vertical position (with tolerance for text line height)
        if detections:
            avg_height = np.mean([d['height'] for d in detections])
            detections.sort(key=lambda d: (round(d['y_center']/avg_height), d['x_center']))
        
        # Combine the sorted text and calculate average confidence
        combined_text = " ".join(d['text'] for d in detections)
        avg_confidence = sum(d['score'] for d in detections) / len(detections)
        
        return combined_text, avg_confidence
        
    except Exception as e:
        print(f"PaddleOCR failed: {e}")
        return "", 0.0

def run_pytesseract(img):
    """Run Tesseract OCR as final fallback"""
    try:
        # Preprocess image specifically for Tesseract
        processed_img = preprocess_for_tesseract(img)
        
        # Configure Tesseract for license plate recognition
        # PSM 8: Treat the image as a single word
        # PSM 7: Treat the image as a single text line
        # PSM 6: Treat the image as a single uniform block of text
        configs = [
            '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ]
        
        best_text = ""
        best_confidence = 0.0
        
        for config in configs:
            try:
                # Get text with confidence data
                data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                
                # Extract text and confidence
                texts = []
                confidences = []
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i])
                    
                    if text and conf > 0:  # Only consider confident detections
                        texts.append(text.upper())
                        confidences.append(conf)
                
                if texts:
                    combined_text = ''.join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    # Clean up the text - remove spaces and non-alphanumeric characters
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', combined_text)
                    
                    if len(cleaned_text) > len(best_text) or (len(cleaned_text) == len(best_text) and avg_confidence > best_confidence):
                        best_text = cleaned_text
                        best_confidence = avg_confidence / 100.0  # Convert to 0-1 range
                        print(f"    Tesseract detected: '{cleaned_text}' (confidence: {avg_confidence:.1f}%) with config: {config}")
                        
            except Exception as e:
                print(f"    Tesseract config failed: {e}")
                continue
        
        return best_text, best_confidence
        
    except Exception as e:
        print(f"Tesseract failed: {e}")
        return "", 0.0

# --- Confidence-Based Voting ---
def ensemble_ocr(img):
    """Combine OCR results using confidence-weighted voting, with Tesseract as final fallback"""
    processed_img = preprocess_plate(img)
    results = []
    
    # Run EasyOCR
    easy_text, easy_conf = run_easyocr(processed_img)
    if easy_text:
        results.append({
            'engine': 'easyocr',
            'text': easy_text,
            'length': len(easy_text),
            'confidence': easy_conf
        })
    
    # Run PaddleOCR
    paddle_text, paddle_conf = run_paddle_ocr(processed_img)
    if paddle_text:
        results.append({
            'engine': 'paddle',
            'text': paddle_text,
            'length': len(paddle_text),
            'confidence': paddle_conf
        })
    
    # If we have results from primary engines, evaluate them
    if results:
        # 1. Check for consensus (same text regardless of confidence)
        if len(results) > 1 and results[0]['text'] == results[1]['text']:
            return results[0]['text']
        
        # 2. Filter valid results (reasonable length and confidence > 0.7)
        valid_results = [r for r in results if 4 <= r['length'] <= 12 and r['confidence'] > 0.7]
        
        if valid_results:
            # Return result with highest confidence
            return max(valid_results, key=lambda x: x['confidence'])['text']
        
        # 3. Relaxed filter: reasonable length but lower confidence threshold
        relaxed_results = [r for r in results if 3 <= r['length'] <= 15 and r['confidence'] > 0.3]
        
        if relaxed_results:
            return max(relaxed_results, key=lambda x: x['confidence'])['text']
        
        # 4. If primary engines have some result but low quality, still check if it's reasonable
        if results:
            best_result = max(results, key=lambda x: (x['length'], x['confidence']))
            if best_result['length'] >= 3:  # At least 3 characters
                return best_result['text']
    
    # 5. FINAL FALLBACK: Run Tesseract if no good results from primary engines
    print("    Running Tesseract as final fallback...")
    tesseract_text, tesseract_conf = run_pytesseract(img)
    
    if tesseract_text:
        results.append({
            'engine': 'tesseract',
            'text': tesseract_text,
            'length': len(tesseract_text),
            'confidence': tesseract_conf
        })
        
        # If Tesseract gives a reasonable result, use it
        if 3 <= len(tesseract_text) <= 15:
            return tesseract_text
    
    # 6. Absolute final fallback: return the best we have, even if it's not great
    if results:
        best_result = max(results, key=lambda x: (x['length'], x['confidence']))
        return best_result['text']
    
    return ""

# --- OCR Processing ---
def perform_ocr(image_path):
    print(f"Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Failed to load image: {image_path}")
        return ""
    
    # Get ensemble result
    result = ensemble_ocr(img)
    print(f"  Final result: {result}")
    return result

# --- Main Processing Loop ---
data = []
preview_images = []
failed_count = 0

print("Starting OCR processing with voting system (including Tesseract fallback)...")

# Process only a few images first for debugging
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for i, image_filename in enumerate(image_files):
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        failed_count += 1
        continue
        
    print(f"\n=== Processing {i+1}: {image_filename} ===")
    ocr_text = perform_ocr(image_path)
    
    # Store results
    data.append({
        'filename': image_filename, 
        'ocr_text': ocr_text,
        'text_length': len(ocr_text)
    })
    
    # For preview
    img = cv2.imread(image_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preview_images.append((img_rgb, image_filename, ocr_text))

# --- Save CSV ---
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"\n✅ OCR finished. Saved to {output_csv}")
print(f"Processed {len(data)} plates, {failed_count} files missing")

# Print statistics
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

# Display results
if preview_images:
    plot_results(preview_images)