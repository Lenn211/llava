"""
Zero-shot inference with YOLO-World (no training required).
Uses text prompts to detect custom objects.

Usage:
    python zero_shot_inference.py --source example_5.jpeg --prompts "wall socket" "power outlet"
    python zero_shot_inference.py --source socket_training/valid/images --prompts "electrical socket" --save-dir results
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from glob import glob
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot detection with YOLO-World")
    p.add_argument('--source', default='IMG20240913124919_jpg.rf.7266fea4a6926c4030ea127ce0196572.jpg', help='Image file or folder to run inference on')
    p.add_argument('--prompts', nargs='+', default=[
        'fluorescent lamp',
        'fluorescent light',
        'tube light',
        'ceiling light',
        'fluorescent tube',
        'overhead light',
        'strip light',
        'light fixture',
        'lamp',
        'lighting'
    ], help='Text prompts describing what to detect (space-separated)')
    p.add_argument('--model', default='yolov8x-world.pt', help='YOLO model weights (yolov8x-world.pt is the largest YOLO-World model available)')
    p.add_argument('--conf', type=float, default=0.05, help='Confidence threshold')
    p.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    p.add_argument('--save-dir', default='fluorescent_lamp_results', help='Directory to save annotated images')
    p.add_argument('--device', default=None, help='Device: 0, cpu, etc.')
    return p.parse_args()


def run_zero_shot(model_path, source, prompts, conf=0.05, imgsz=640, save_dir='fluorescent_lamp_results', device=None):
    """
    Run YOLO-World zero-shot detection with text prompts.
    """
    # Auto-detect GPU if device not specified
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading YOLO-World model: {model_path}")
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Using device: {device}")
    else:
        print("⚠️ GPU not available, using CPU")
    
    model = YOLO(model_path)
    
    # Set custom classes via text prompts (only works with YOLO-World models)
    if hasattr(model, 'set_classes') or 'world' in model_path.lower():
        print(f"Setting detection prompts: {prompts}")
        if hasattr(model, 'set_classes'):
            model.set_classes(prompts)
    else:
        print(f"⚠️ Standard YOLO model detected - using pre-trained COCO classes")
        print(f"   For custom prompts, use a YOLO-World model (e.g., yolov8l-world.pt)")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get list of images
    if os.path.isdir(source):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob(os.path.join(source, ext)))
        image_files.extend(glob(os.path.join(source, '**', '*.jpg'), recursive=True))
        image_files.extend(glob(os.path.join(source, '**', '*.jpeg'), recursive=True))
        image_files.extend(glob(os.path.join(source, '**', '*.png'), recursive=True))
        image_files = list(set(image_files))  # Remove duplicates
    else:
        image_files = [source]
    
    print(f"Found {len(image_files)} images to process")
    
    detection_count = 0
    for img_path in image_files:
        print(f"Processing: {img_path}")
        
        # Run prediction
        results = model.predict(source=img_path, conf=conf, imgsz=imgsz, device=device, verbose=False)
        
        # Load image for annotation
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not load {img_path}")
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
        img_name = Path(img_path).name
        output_path = os.path.join(save_dir, f"annotated_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"  Detections: {num_detections} | Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Processed {len(image_files)} images")
    print(f"Total detections: {detection_count}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    
    # Model will be downloaded automatically if not found
    if not os.path.exists(args.model):
        print(f"⚠️ Model file '{args.model}' not found - will download automatically...")
    
    if not os.path.exists(args.source):
        print(f"Error: Source '{args.source}' not found.")
        return
    
    run_zero_shot(
        model_path=args.model,
        source=args.source,
        prompts=args.prompts,
        conf=args.conf,
        imgsz=args.imgsz,
        save_dir=args.save_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
