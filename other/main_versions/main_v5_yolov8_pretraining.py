import cv2
from ultralytics import YOLO
import os
import glob
from PIL import Image

# Path to custom dataset YAML (Roboflow export)
dataset_yaml = "socket_training/data.yaml"
custom_weights = "yolov8l_power_switch.pt"  # Using yolov8l-world

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

# Train only if custom weights do not exist
if not os.path.exists(custom_weights):
    print("Starting YOLOv8l-world training on power sockets...")
    print("⚠️ Using aggressive memory optimization settings...")
    
    model = YOLO("yolov8l-world.pt")  # Load pre-trained YOLO-World model
    
    # Extremely aggressive memory optimization
    results = model.train(
        data=dataset_yaml, 
        epochs=10, 
        imgsz=256,  # Reduced from 320 to 256
        batch=4,    # Reduced from 8 to 4
        name="yolov8l_power_switch",
        patience=5,
        cache=False,  # Don't cache images
        workers=4,    # Reduced to 4 workers
        amp=True,     # Use automatic mixed precision (FP16)
        device='cpu'  # Force CPU training to avoid GPU memory issues
    )
    
    # Save best weights
    best_weight_path = os.path.join("runs/detect/yolov8l_power_switch/weights/best.pt")
    if os.path.exists(best_weight_path):
        os.rename(best_weight_path, custom_weights)
        print(f"Custom weights saved as {custom_weights}")
    else:
        # Try to find the latest run directory with best.pt
        run_dirs = glob.glob("runs/detect/yolov8l_power_switch*/weights/best.pt")
        if run_dirs:
            latest_best = sorted(run_dirs, key=os.path.getmtime)[-1]
            os.rename(latest_best, custom_weights)
            print(f"Custom weights saved as {custom_weights} from {latest_best}")
        else:
            print("Training finished, but best weights not found.")
else:
    print(f"Custom weights {custom_weights} already exist.")

# Load trained model for inference
model = YOLO(custom_weights)

# --- Detection functions ---
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

# --- Inference on example image ---
image = cv2.imread("example_3.png")
if image is None:
    raise FileNotFoundError("Image 'example_4.png' not found or could not be loaded.")

result_img, _ = predict_and_detect(model, image, classes=[], conf=0.1)
print(f"Type of result_img: {type(result_img)}")  # Debug: check type
cv2.imwrite("annotated_results.jpg", result_img)
cv2.waitKey(0)