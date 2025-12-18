import cv2
from ultralytics import YOLO
import os
import glob
from PIL import Image
import torch
import numpy as np
from collections import defaultdict

# Define inspection prompts with synonyms for better detection
INSPECTION_ELEMENTS = {
    "sockets": ["wall socket", "power outlet", "electrical socket", "power point", "wall outlet"],
    "fire_safety": ["fire extinguisher", "fire safety equipment", "fire suppression device"],
    "lighting": ["light fixture", "ceiling light", "wall light", "lamp", "light fitting"]
}

def get_element_category(detected_label):
    """Map a detected label back to its category"""
    detected_label = detected_label.lower()
    for category, prompts in INSPECTION_ELEMENTS.items():
        if any(prompt.lower() in detected_label for prompt in prompts):
            return category
    return None

def detect_in_image(model, image_path, save_dir='batch_detection_results', device=0):
    """Run detection on a single image and return categorized results"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Run prediction with GPU device
    results = model.predict(source=image_path, conf=0.10, verbose=False, device=device)
    
    # Store detections by category
    category_detections = {
        "sockets": [],
        "fire_safety": [],
        "lighting": []
    }
    
    # Process results
    if results and len(results) > 0:
        result = results[0]
        img = cv2.imread(image_path)
        
        for box in result.boxes:
            # Get detection info
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            
            # Categorize detection
            category = get_element_category(class_name)
            if category:
                category_detections[category].append({
                    "label": class_name,
                    "confidence": float(confidence),
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })
                
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                cv2.putText(img, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save annotated image
        output_path = os.path.join(save_dir, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, img)
    
    return category_detections

def main():
    # Source folder to inspect
    images_folder = "examples"
    if not os.path.exists(images_folder):
        print(f"Error: Folder {images_folder} not found")
        return
    
    # Get all image files in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
    
    if not image_files:
        print(f"No images found in {images_folder}")
        return
    
    print(f"🔍 Starting batch detection analysis on {len(image_files)} images...")
    print("=" * 80)
    
    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Using device: GPU (device {device})")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    else:
        print("⚠️ GPU not available, using CPU")
    
    # Define custom model path
    custom_model = "yolov8l_power_switch.pt"
    
    # Check if custom trained model exists
    if os.path.exists(custom_model):
        print(f"📋 Loading custom-trained model: {custom_model}")
        model = YOLO(custom_model)
        print(f"✅ Custom model loaded (trained on power sockets)")
        use_custom = True
    else:
        print(f"⚠️ Custom model not found: {custom_model}")
        print(f"📋 Loading pre-trained YOLO-World model instead...")
        model = YOLO("yolov8l-world.pt")
        
        # Combine all prompts for detection (only needed for YOLO-World)
        all_prompts = []
        for prompts in INSPECTION_ELEMENTS.values():
            all_prompts.extend(prompts)
        model.set_classes(all_prompts)
        print(f"✅ Model loaded. Scanning for: {', '.join(INSPECTION_ELEMENTS.keys())}")
        use_custom = False
    
    print("=" * 80)
    
    # Store all results
    all_results = []
    
    # Track confidence scores per category
    category_confidences = {
        "sockets": [],
        "fire_safety": [],
        "lighting": []
    }
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_name}")
        
        # Detect elements in the image with GPU
        detections = detect_in_image(model, image_path, device=device)
        
        # Record results for this image
        image_results = {
            "image": image_name,
            "detections": {}
        }
        
        # Process each category
        for category in INSPECTION_ELEMENTS.keys():
            detected_items = detections[category]
            
            if detected_items:
                # Get the highest confidence detection for this category
                max_confidence = max(item["confidence"] for item in detected_items)
                category_confidences[category].append(max_confidence)
                
                image_results["detections"][category] = {
                    "count": len(detected_items),
                    "max_confidence": max_confidence,
                    "items": detected_items
                }
                
                print(f"  ✅ {category}: {len(detected_items)} detected (max confidence: {max_confidence:.2f})")
            else:
                # No detection, record 0.00 confidence
                category_confidences[category].append(0.00)
                image_results["detections"][category] = {
                    "count": 0,
                    "max_confidence": 0.00,
                    "items": []
                }
                print(f"  ❌ {category}: Not detected (confidence: 0.00)")
        
        all_results.append(image_results)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clean up model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Calculate and display statistics
    print("\n" + "=" * 80)
    print("📊 DETECTION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal images processed: {len(image_files)}")
    
    # Overall statistics per category
    for category in INSPECTION_ELEMENTS.keys():
        confidences = category_confidences[category]
        avg_confidence = np.mean(confidences) if confidences else 0.00
        detected_count = sum(1 for c in confidences if c > 0)
        detection_rate = (detected_count / len(confidences) * 100) if confidences else 0
        
        print(f"\n{category.upper()}:")
        print(f"  Average Confidence: {avg_confidence:.2f}")
        print(f"  Detection Rate: {detection_rate:.1f}% ({detected_count}/{len(confidences)} images)")
        print(f"  Min Confidence: {min(confidences):.2f}")
        print(f"  Max Confidence: {max(confidences):.2f}")
    
    # Overall average across all categories
    all_confidences = []
    for confidences in category_confidences.values():
        all_confidences.extend(confidences)
    
    overall_avg = np.mean(all_confidences) if all_confidences else 0.00
    print(f"\n{'=' * 80}")
    print(f"OVERALL AVERAGE CONFIDENCE: {overall_avg:.2f}")
    print(f"{'=' * 80}")
    
    # Save detailed report to file
    report_path = "batch_detection_results/detection_report.txt"
    with open(report_path, 'w') as f:
        f.write("BATCH DETECTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total images processed: {len(image_files)}\n\n")
        
        # Per-image details
        f.write("DETAILED RESULTS PER IMAGE:\n")
        f.write("-" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"Image: {result['image']}\n")
            for category in INSPECTION_ELEMENTS.keys():
                det = result['detections'][category]
                f.write(f"  {category}: {det['count']} detected, confidence: {det['max_confidence']:.2f}\n")
            f.write("\n")
        
        # Summary statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        for category in INSPECTION_ELEMENTS.keys():
            confidences = category_confidences[category]
            avg_confidence = np.mean(confidences) if confidences else 0.00
            detected_count = sum(1 for c in confidences if c > 0)
            detection_rate = (detected_count / len(confidences) * 100) if confidences else 0
            
            f.write(f"{category.upper()}:\n")
            f.write(f"  Average Confidence: {avg_confidence:.2f}\n")
            f.write(f"  Detection Rate: {detection_rate:.1f}% ({detected_count}/{len(confidences)} images)\n")
            f.write(f"  Min Confidence: {min(confidences):.2f}\n")
            f.write(f"  Max Confidence: {max(confidences):.2f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"OVERALL AVERAGE CONFIDENCE: {overall_avg:.2f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    print(f"📁 Annotated images saved to: batch_detection_results/")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
