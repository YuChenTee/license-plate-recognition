from difflib import SequenceMatcher

import pandas as pd


def load_csv(path, is_ground_truth=False):
    """Load CSV and optionally filter empty OCR results."""
    df = pd.read_csv(path)
    df["filename"] = df["filename"].astype(str)
    df["ocr_text"] = df["ocr_text"].astype(str).str.strip()  # Now using 'ocr_text'

    # Only filter empty entries if loading ground truth
    if is_ground_truth:
        df = df[df["ocr_text"] != "nan"]

    return df[["filename", "ocr_text"]]


def normalize(text):
    """Normalize text for comparison:
    - Convert to uppercase
    - Remove all spaces and special characters
    - Only allow alphanumeric characters (0-9, A-Z).
    """
    allowed_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # Convert to string, uppercase, and filter allowed characters
    normalized = "".join(c for c in str(text).upper() if c in allowed_chars)
    return normalized


def compare_ocr(gt_df, ocr_df, label):
    """Compare OCR results with ground truth, skipping empty GT entries."""
    # Merge dataframes on filename (inner join to skip missing filenames)
    merged = gt_df.merge(ocr_df, on="filename", how="inner", suffixes=("_gt", f"_{label}"))

    # Normalize text
    merged["ocr_text_gt_norm"] = merged["ocr_text_gt"].apply(normalize)
    merged[f"ocr_text_{label}_norm"] = merged[f"ocr_text_{label}"].apply(normalize)

    # Calculate metrics
    exact_matches = (merged["ocr_text_gt_norm"] == merged[f"ocr_text_{label}_norm"]).sum()
    total = len(merged)
    accuracy = exact_matches / total if total > 0 else 0

    similarities = [
        SequenceMatcher(None, gt, pred).ratio()
        for gt, pred in zip(merged["ocr_text_gt_norm"], merged[f"ocr_text_{label}_norm"])
    ]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0

    # Print results
    print(f"📊 {label} OCR Results")
    print(f"- Total Samples: {total}")
    print(f"- Exact Match Accuracy: {accuracy * 100:.2f}%")
    print(f"- Average Similarity: {avg_similarity * 100:.2f}%")
    print("-" * 40)

    return merged


if __name__ == "__main__":
    labels_path = "labels.csv"
    pytess_path = "labels_pytesseract.csv"
    easyocr_path = "labels_easyocr.csv"
    paddleocr_path = "labels_paddleocr.csv"  # Assuming you have a PaddleOCR CSV
    combined_path = "labels_easyocr_improved.csv"

    # Load data
    gt_df = load_csv(labels_path, is_ground_truth=True).rename(columns={"ocr_text": "ocr_text_gt"})
    tess_df = load_csv(pytess_path).rename(columns={"ocr_text": "ocr_text_pytesseract"})
    easy_df = load_csv(easyocr_path).rename(columns={"ocr_text": "ocr_text_easyocr"})
    paddle_df = load_csv(paddleocr_path).rename(columns={"ocr_text": "ocr_text_paddleocr"})
    combined_df = load_csv(combined_path).rename(columns={"ocr_text": "ocr_text_improved"})

    compare_ocr(gt_df, tess_df, "pytesseract")
    compare_ocr(gt_df, easy_df, "easyocr")
    compare_ocr(gt_df, paddle_df, "paddleocr")
    compare_ocr(gt_df, combined_df, "improved")

    # Merge all on filename
    df_all = (
        gt_df.merge(tess_df, on="filename", how="inner")
        .merge(easy_df, on="filename", how="inner")
        .merge(paddle_df, on="filename", how="inner")
        .merge(combined_df, on="filename", how="inner")
    )

    # Normalize columns
    df_all["ocr_text_gt_norm"] = df_all["ocr_text_gt"].apply(normalize)
    df_all["ocr_text_pytesseract_norm"] = df_all["ocr_text_pytesseract"].apply(normalize)
    df_all["ocr_text_easyocr_norm"] = df_all["ocr_text_easyocr"].apply(normalize)
    df_all["ocr_text_paddleocr_norm"] = df_all["ocr_text_paddleocr"].apply(normalize)
    df_all["ocr_text_improved_norm"] = df_all["ocr_text_improved"].apply(normalize)

    # Optional: Add match and similarity scores
    def get_score(gt, pred):
        return SequenceMatcher(None, gt, pred).ratio()

    df_all["match_pytesseract"] = df_all["ocr_text_gt_norm"] == df_all["ocr_text_pytesseract_norm"]
    df_all["similarity_pytesseract"] = df_all.apply(
        lambda row: get_score(row["ocr_text_gt_norm"], row["ocr_text_pytesseract_norm"]), axis=1
    )

    df_all["match_easyocr"] = df_all["ocr_text_gt_norm"] == df_all["ocr_text_easyocr_norm"]
    df_all["similarity_easyocr"] = df_all.apply(
        lambda row: get_score(row["ocr_text_gt_norm"], row["ocr_text_easyocr_norm"]), axis=1
    )

    df_all["match_paddleocr"] = df_all["ocr_text_gt_norm"] == df_all["ocr_text_paddleocr_norm"]
    df_all["similarity_paddleocr"] = df_all.apply(
        lambda row: get_score(row["ocr_text_gt_norm"], row["ocr_text_paddleocr_norm"]), axis=1
    )

    df_all["match_improved"] = df_all["ocr_text_gt_norm"] == df_all["ocr_text_improved_norm"]
    df_all["similarity_improved"] = df_all.apply(
        lambda row: get_score(row["ocr_text_gt_norm"], row["ocr_text_improved_norm"]), axis=1
    )

    # Export everything
    df_all.to_csv("ocr_all_comparison.csv", index=False)
    print("✅ Merged comparison saved to ocr_all_comparison.csv")
