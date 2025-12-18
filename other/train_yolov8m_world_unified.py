"""
Train YOLOv8m-world model on merged multi-class dataset.
OPTIMIZED FOR LOW VRAM (4GB GPUs like Quadro T1000)

Uses YOLOv8m-world (Medium) instead of YOLOv8x-world (Extra Large)
- 63% smaller model
- Much lower GPU memory requirements
- Still very good accuracy

This script:
1. Merges all datasets into a unified structure
2. Relabels classes to: 'fluorescent tube', 'fire extinguisher', 'outlet'
3. Trains YOLOv8m-world model (optimized for 4GB VRAM)
"""

import cv2
from ultralytics import YOLO
import os
import glob
import shutil
from pathlib import Path
import torch
import yaml

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def create_unified_dataset():
    """
    Merge all datasets and relabel classes uniformly:
    - Fluorescent lamp 1 & 2 -> 'fluorescent tube' (class 0)
    - Fire extinguisher -> 'fire extinguisher' (class 1)  
    - Socket 2 -> 'outlet' (class 2)
    """
    
    # Define unified dataset structure
    unified_dataset_dir = "unified_dataset"
    train_images_dir = os.path.join(unified_dataset_dir, "train", "images")
    train_labels_dir = os.path.join(unified_dataset_dir, "train", "labels")
    val_images_dir = os.path.join(unified_dataset_dir, "valid", "images")
    val_labels_dir = os.path.join(unified_dataset_dir, "valid", "labels")
    
    # Create directories
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("=" * 60)
    print("📦 CREATING UNIFIED DATASET")
    print("=" * 60)
    
    # Class mapping: all classes map to their unified class IDs
    # fluorescent tube -> 0, fire extinguisher -> 1, outlet -> 2
    
    datasets_info = [
        {
            'name': 'Fluorescent Lamp 1',
            'path': 'flourescent lamp 1',
            'new_class_id': 0,
            'new_class_name': 'fluorescent tube'
        },
        {
            'name': 'Fluorescent Lamp 2',
            'path': 'fluorescent lamp 2',
            'new_class_id': 0,
            'new_class_name': 'fluorescent tube'
        },
        {
            'name': 'Fire Extinguisher',
            'path': 'fire extinguisher',
            'new_class_id': 1,
            'new_class_name': 'fire extinguisher'
        },
        {
            'name': 'Socket 2',
            'path': 'socket 2',
            'new_class_id': 2,
            'new_class_name': 'outlet'
        }
    ]
    
    total_train_images = 0
    total_val_images = 0
    
    for dataset_info in datasets_info:
        dataset_path = dataset_info['path']
        new_class_id = dataset_info['new_class_id']
        dataset_name = dataset_info['name']
        
        print(f"\n📂 Processing: {dataset_name}")
        print(f"   Relabeling all objects as: '{dataset_info['new_class_name']}' (class {new_class_id})")
        
        if not os.path.exists(dataset_path):
            print(f"   ⚠️ Dataset not found: {dataset_path}")
            continue
        
        # Process train and validation splits
        for split in ['train', 'valid']:
            src_images = os.path.join(dataset_path, split, 'images')
            src_labels = os.path.join(dataset_path, split, 'labels')
            
            if not os.path.exists(src_images):
                print(f"   ⚠️ {split} images not found: {src_images}")
                continue
            
            # Get destination directories
            if split == 'train':
                dst_images = train_images_dir
                dst_labels = train_labels_dir
            else:
                dst_images = val_images_dir
                dst_labels = val_labels_dir
            
            # Copy and relabel
            image_files = glob.glob(os.path.join(src_images, '*.jpg')) + \
                         glob.glob(os.path.join(src_images, '*.jpeg')) + \
                         glob.glob(os.path.join(src_images, '*.png'))
            
            copied_count = 0
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                # Create unique filename to avoid conflicts
                unique_name = f"{dataset_info['new_class_name'].replace(' ', '_')}_{Path(dataset_path).name.replace(' ', '_')}_{img_name}"
                
                # Copy image
                dst_img_path = os.path.join(dst_images, unique_name)
                shutil.copy(img_path, dst_img_path)
                
                # Process label file
                label_name = os.path.splitext(img_name)[0] + '.txt'
                src_label_path = os.path.join(src_labels, label_name)
                dst_label_path = os.path.join(dst_labels, os.path.splitext(unique_name)[0] + '.txt')
                
                if os.path.exists(src_label_path):
                    # Read and relabel
                    with open(src_label_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Rewrite with new class ID
                    with open(dst_label_path, 'w') as f:
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                # Replace old class ID with new unified class ID
                                parts[0] = str(new_class_id)
                                f.write(' '.join(parts) + '\n')
                    
                    copied_count += 1
            
            if split == 'train':
                total_train_images += copied_count
            else:
                total_val_images += copied_count
            
            print(f"   ✅ {split}: {copied_count} images")
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset merge complete!")
    print(f"   Total training images: {total_train_images}")
    print(f"   Total validation images: {total_val_images}")
    print(f"{'='*60}\n")
    
    # Create unified data.yaml
    data_yaml = {
        'train': os.path.abspath(train_images_dir),
        'val': os.path.abspath(val_images_dir),
        'nc': 3,
        'names': ['fluorescent tube', 'fire extinguisher', 'outlet']
    }
    
    yaml_path = os.path.join(unified_dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"📄 Created unified data.yaml: {yaml_path}")
    print(f"   Classes: {data_yaml['names']}\n")
    
    return yaml_path


def train_yolov8m_world(data_yaml, epochs=100, batch=4, imgsz=416):
    """
    Train YOLOv8m-world model on unified dataset.
    OPTIMIZED FOR 4GB VRAM (Quadro T1000)
    """
    
    model_path = "yolov8m-world.pt"  # Medium model (63% smaller than x-world)
    custom_weights = "yolov8m_world_custom_trained.pt"
    
    # Auto-detect GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("🚀 STARTING YOLOV8M-WORLD TRAINING (LOW VRAM OPTIMIZED)")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU Memory: {total_memory:.2f} GB")
        print(f"   Using device: {device}")
    else:
        print("⚠️ GPU not available, using CPU (training will be VERY slow)")
        print("   Recommend reducing batch size and epochs for CPU training")
    
    print(f"\n📋 Training Configuration (Optimized for Low VRAM):")
    print(f"   Base model: {model_path} (YOLOv8m-world - Medium, 25M params)")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch} (small for 4GB VRAM)")
    print(f"   Image size: {imgsz} (reduced for memory)")
    print(f"   Device: {device}")
    print(f"   Memory optimizations: AMP enabled, no caching")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train with aggressive memory optimization
    print("Starting training...\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="yolov8m_world_custom",
        patience=20,        # Early stopping patience
        cache=False,        # Don't cache images in RAM
        workers=1,          # Reduced workers for low memory
        device=device,
        amp=True,           # Automatic Mixed Precision (saves ~40% GPU memory)
        lr0=0.01,           # Initial learning rate
        lrf=0.01,           # Final learning rate
        momentum=0.937,     # SGD momentum
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        plots=True,
        save=True,
        save_period=10,     # Save checkpoint every 10 epochs
        val=True,
        verbose=True,
        fraction=1.0        # Use full dataset
    )
    
    # Save best weights
    best_weight_path = os.path.join("runs/detect/yolov8m_world_custom/weights/best.pt")
    if os.path.exists(best_weight_path):
        shutil.copy(best_weight_path, custom_weights)
        print(f"\n✅ Custom weights saved as {custom_weights}")
    else:
        # Try to find the latest run directory
        run_dirs = glob.glob("runs/detect/yolov8m_world_custom*/weights/best.pt")
        if run_dirs:
            latest_best = sorted(run_dirs, key=os.path.getmtime)[-1]
            shutil.copy(latest_best, custom_weights)
            print(f"\n✅ Custom weights saved as {custom_weights} from {latest_best}")
        else:
            print("\n⚠️ Training finished, but best weights not found.")
            custom_weights = None
    
    return custom_weights


def run_inference(model_path, examples_dir="examples", output_dir="unified_model_results"):
    """
    Run inference on example images using trained model.
    """
    
    print("\n" + "=" * 60)
    print("🔍 RUNNING INFERENCE")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(examples_dir, ext)))
    
    if not image_files:
        print(f"⚠️ No images found in {examples_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Process each image
    total_detections = 0
    for idx, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        print(f"[{idx}/{len(image_files)}] Processing: {img_name}")
        
        # Run detection
        results = model.predict(source=img_path, conf=0.25, imgsz=640, device=device, verbose=False)
        
        # Load image for annotation
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️ Could not load {img_path}")
            continue
        
        # Draw detections
        num_detections = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                cv2.putText(img, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                num_detections += 1
                total_detections += 1
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"annotated_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"  ✅ Detected {num_detections} object(s) | Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Inference complete!")
    print(f"   Processed {len(image_files)} images")
    print(f"   Total detections: {total_detections}")
    print(f"   Results saved to: {output_dir}/")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8m-world on unified multi-class dataset (LOW VRAM)")
    parser.add_argument('--skip-merge', action='store_true', help='Skip dataset merging (use existing unified_dataset)')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing weights)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size (default: 4 for 4GB VRAM)')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size (default: 416 for low VRAM)')
    parser.add_argument('--weights', default='yolov8m_world_custom_trained.pt', help='Path to trained weights for inference')
    
    args = parser.parse_args()
    
    # Step 1: Create unified dataset
    if not args.skip_merge:
        data_yaml = create_unified_dataset()
    else:
        data_yaml = "unified_dataset/data.yaml"
        print(f"⏭️ Skipping dataset merge, using existing: {data_yaml}\n")
    
    # Step 2: Train model
    if not args.skip_training:
        custom_weights = train_yolov8m_world(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz
        )
    else:
        custom_weights = args.weights
        print(f"⏭️ Skipping training, using existing weights: {custom_weights}\n")
    
    # Step 3: Run inference
    if custom_weights and os.path.exists(custom_weights):
        run_inference(model_path=custom_weights)
    else:
        print("\n⚠️ No trained weights available for inference")


if __name__ == '__main__':
    main()
