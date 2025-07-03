import os
from pathlib import Path
import subprocess

# Paths
data_yaml = "../data.yaml"  # Update path if needed
weights_init = "yolov5s.pt"
run_name = "lp-detector"
image_size = 320
batch_size = 16
epochs = 50

# Train command
train_cmd = [
    "python", "train.py",
    "--img", str(image_size),
    "--batch", str(batch_size),
    "--epochs", str(epochs),
    "--data", data_yaml,
    "--weights", weights_init,
    "--name", run_name
]

# Evaluate command (after training)
eval_cmd = [
    "python", "val.py",
    "--data", data_yaml,
    "--weights", f"runs/train/{run_name}/weights/best.pt",
]

def run_command(cmd, description="Running command"):
    print(f"\n🔧 {description}:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    print("🚀 Starting training and evaluation for license plate detection...")
    run_command(train_cmd, "Training YOLOv5 on custom LP dataset")
    # run_command(eval_cmd, "Evaluating best model")
    
    # Confirm where best weights are saved
    best_weights_path = Path(f"runs/train/{run_name}/weights/best.pt")
    if best_weights_path.exists():
        print(f"\n✅ Training completed. Best weights saved to: {best_weights_path.resolve()}")
    else:
        print("\n❌ Training finished but best.pt was not found. Check run logs.")