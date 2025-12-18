"""
Batch zero-shot inference with YOLO-World (no training required).
Processes multiple images from a folder and saves all annotated results.

Usage:
    python batch_zero_shot_inference.py
    python batch_zero_shot_inference.py --prompts "wall socket" "power outlet"
    python batch_zero_shot_inference.py --model yolov8l_power_switch.pt
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from glob import glob
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Batch zero-shot detection with YOLO-World")
    p.add_argument('--source', default='examples', help='Folder containing images to process')
    p.add_argument('--prompts', nargs='+', default=['wall socket', 'power outlet', 'electrical socket'], 
                   help='Text prompts describing what to detect (space-separated)')
    p.add_argument('--model', default='yolov8n_power_switch_finetuned.pt', help='YOLO model weights (YOLO-World or custom trained)')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    p.add_argument('--save-dir', default='batch_finetuned_results', help='Directory to save all annotated images')
    p.add_argument('--device', default=None, help='Device: 0 (GPU), cpu, etc.')
    return p.parse_args()


def run_batch_zero_shot(model_path, source_folder, prompts, conf=0.25, imgsz=640, save_dir='batch_finetuned_results', device=None):
    """
    Run YOLO-World zero-shot detection on all images in a folder.
    """
    # Auto-detect GPU if device not specified
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("BATCH ZERO-SHOT OBJECT DETECTION")
    print("=" * 80)
    
    # Display GPU/CPU info
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Using device: GPU (device {device})")
    else:
        print("⚠️ GPU not available, using CPU")
    
    # Check if using custom trained model or YOLO-World
    is_custom_model = 'power_switch' in model_path or not 'world' in model_path.lower()
    
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    if is_custom_model:
        print(f"✅ Custom trained model loaded")
        print(f"   Note: Custom models use their trained classes, prompts are ignored")
    else:
        # Set custom classes via text prompts (only for YOLO-World)
        print(f"✅ YOLO-World model loaded")
        print(f"   Setting detection prompts: {prompts}")
        model.set_classes(prompts)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get list of all images from source folder
    if not os.path.isdir(source_folder):
        print(f"Error: '{source_folder}' is not a valid folder")
        return
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob(os.path.join(source_folder, ext)))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"Error: No images found in '{source_folder}'")
        return
    
    print(f"\nFound {len(image_files)} images to process from '{source_folder}'")
    print("=" * 80)
    
    detection_count = 0
    processed_count = 0
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_name}")
        
        # Run prediction
        results = model.predict(source=img_path, conf=conf, imgsz=imgsz, device=device, verbose=False)
        
        # Load image for annotation
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️ Warning: Could not load {img_path}")
            continue
        
        # Draw detections
        num_detections = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10), 
                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                cv2.putText(img, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                num_detections += 1
                detection_count += 1
        
        # Save annotated image
        output_filename = f"annotated_{img_name}"
        output_path = os.path.join(save_dir, output_filename)
        cv2.imwrite(output_path, img)
        
        if num_detections > 0:
            print(f"  ✅ Detections: {num_detections}")
        else:
            print(f"  ❌ No detections")
        print(f"  💾 Saved: {output_path}")
        
        processed_count += 1
        
        # Clear GPU cache after each image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Processed: {processed_count}/{len(image_files)} images")
    print(f"✅ Total detections: {detection_count}")
    print(f"✅ Results saved to: {save_dir}/")
    print("=" * 80)
    
    # List all saved files
    print(f"\nAnnotated images:")
    saved_files = sorted(glob(os.path.join(save_dir, "annotated_*")))
    for i, f in enumerate(saved_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    print(f"\n💡 Tip: Check the '{save_dir}' folder to view all annotated results!")


def main():
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Available options:")
        print("  - yolov8n_power_switch_finetuned.pt (YOLOv8n fine-tuned model)")
        print("  - yolov8l-world.pt (zero-shot YOLO-World model)")
        print("  - yolov8l_power_switch.pt (custom trained model)")
        return
    
    # Check if source folder exists
    if not os.path.exists(args.source):
        print(f"Error: Source folder '{args.source}' not found.")
        return
    
    run_batch_zero_shot(
        model_path=args.model,
        source_folder=args.source,
        prompts=args.prompts,
        conf=args.conf,
        imgsz=args.imgsz,
        save_dir=args.save_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
