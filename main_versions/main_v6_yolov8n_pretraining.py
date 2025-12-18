import cv2
from ultralytics import YOLO
import os
import glob
from PIL import Image
import torch

# Path to custom dataset YAML (Roboflow export)
dataset_yaml = "socket_training/data.yaml"
custom_weights = "yolov8n_power_switch_finetuned.pt"  # Using yolov8n base model

# Ensure validation images folder exists and has images
val_images_dir = "socket_training/valid/images"
os.makedirs(val_images_dir, exist_ok=True)

# If validation folder is empty, copy 5 images from train/images
if not os.listdir(val_images_dir):
    train_images_dir = "socket_training/train/images"
    train_labels_dir = "socket_training/train/labels"
    val_labels_dir = "socket_training/valid/labels"
    os.makedirs(val_labels_dir, exist_ok=True)
    train_images = [f for f in os.listdir(train_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for img_name in train_images[:5]:
        src_img = os.path.join(train_images_dir, img_name)
        dst_img = os.path.join(val_images_dir, img_name)
        if not os.path.exists(dst_img):
            import shutil
            shutil.copy(src_img, dst_img)
        # Copy corresponding label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(train_labels_dir, label_name)
        dst_label = os.path.join(val_labels_dir, label_name)
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            shutil.copy(src_label, dst_label)
    print(f"Copied {min(5, len(train_images))} images and labels to validation folder.")

# Force CPU usage
device = 'cpu'
if torch.cuda.is_available():
    print(f"ℹ️ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   But forcing CPU for training (device={device})")
else:
    print("⚠️ GPU not available, using CPU")

# Train only if custom weights do not exist
if not os.path.exists(custom_weights):
    print("Starting YOLOv8n fine-tuning on power sockets...")
    print("📋 Base model: YOLOv8n (nano - fastest and smallest)")
    
    model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8n model (nano version)
    
    # Training with CPU and optimized settings for YOLOv8n
    results = model.train(
        data=dataset_yaml, 
        epochs=20,          # More epochs for better fine-tuning
        imgsz=256,          # Standard image size for YOLOv8n
        batch=16,           # Larger batch size (YOLOv8n is smaller)
        name="yolov8n_power_switch_finetuned",
        patience=10,        # Early stopping patience
        cache=False,        # Don't cache images in RAM
        workers=4,          # Number of dataloader workers
        amp=False,          # Disable AMP for CPU (only works on GPU)
        device='cpu',       # Force CPU training
        lr0=0.01,           # Initial learning rate
        lrf=0.01,           # Final learning rate (lr0 * lrf)
        momentum=0.937,     # SGD momentum
        weight_decay=0.0005, # Optimizer weight decay
        warmup_epochs=3,    # Warmup epochs
        warmup_momentum=0.8, # Warmup initial momentum
        box=7.5,            # Box loss gain
        cls=0.5,            # Class loss gain
        dfl=1.5,            # DFL loss gain
        plots=True,         # Save training plots
        save=True,          # Save checkpoints
        save_period=-1,     # Save checkpoint every x epochs (disabled with -1)
        val=True,           # Validate during training
        verbose=True        # Verbose output
    )
    
    # Save best weights
    best_weight_path = os.path.join("runs/detect/yolov8n_power_switch_finetuned/weights/best.pt")
    if os.path.exists(best_weight_path):
        import shutil
        shutil.copy(best_weight_path, custom_weights)
        print(f"✅ Custom weights saved as {custom_weights}")
    else:
        # Try to find the latest run directory with best.pt
        run_dirs = glob.glob("runs/detect/yolov8n_power_switch_finetuned*/weights/best.pt")
        if run_dirs:
            latest_best = sorted(run_dirs, key=os.path.getmtime)[-1]
            import shutil
            shutil.copy(latest_best, custom_weights)
            print(f"✅ Custom weights saved as {custom_weights} from {latest_best}")
        else:
            print("⚠️ Training finished, but best weights not found.")
else:
    print(f"✅ Custom weights {custom_weights} already exist. Skipping training.")

# Load trained model for inference
print(f"\n📋 Loading fine-tuned model: {custom_weights}")
model = YOLO(custom_weights)

# --- Detection functions ---
def predict(chosen_model, img, classes=[], conf=0.5, device='cpu'):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, device=device)
    else:
        results = chosen_model.predict(img, conf=conf, device=device)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, device='cpu'):
    results = predict(chosen_model, img, classes, conf=conf, device=device)
    for result in results:
        for box in result.boxes:
            # Draw bounding box
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
            # Draw label with confidence
            label = f"{result.names[int(box.cls[0])]}: {float(box.conf[0]):.2f}"
            cv2.putText(img, label,
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img, results

# --- Batch inference on all example images ---
print("\n🔍 Running inference on example images...")
examples_dir = "examples"
output_dir = "finetuned_yolov8n_results"
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(examples_dir, ext)))

if not image_files:
    print(f"⚠️ No images found in {examples_dir}")
    # Fallback to a single image
    test_image = "examples/example_1.jpg"
    if os.path.exists(test_image):
        image_files = [test_image]
    else:
        print("⚠️ No example images found. Skipping inference.")
        exit(0)

print(f"Found {len(image_files)} images to process")

# Process each image
for idx, img_path in enumerate(image_files, 1):
    img_name = os.path.basename(img_path)
    print(f"\n[{idx}/{len(image_files)}] Processing: {img_name}")
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"  ⚠️ Could not load {img_path}")
        continue
    
    # Run detection with CPU
    result_img, results = predict_and_detect(model, image.copy(), classes=[], conf=0.25, device='cpu')
    
    # Count detections
    num_detections = sum(len(result.boxes) for result in results)
    print(f"  ✅ Detected {num_detections} object(s)")
    
    # Save annotated image
    output_path = os.path.join(output_dir, f"annotated_{img_name}")
    cv2.imwrite(output_path, result_img)
    print(f"  💾 Saved: {output_path}")

print(f"\n{'='*60}")
print(f"✅ Batch inference complete!")
print(f"📁 Results saved to: {output_dir}/")
print(f"{'='*60}")
