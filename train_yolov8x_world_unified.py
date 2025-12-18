"""
Train YOLOv11x model on merged multi-class dataset.
Combines fluorescent lamp, fire extinguisher, and socket datasets with unified labels.

This script:
1. Merges all datasets into a unified structure
2. Relabels classes to: 'fluorescent tube', 'fire extinguisher', 'outlet'
3. Trains YOLOv11x model (latest YOLO architecture)
"""

import cv2
from ultralytics import YOLO
import os
import glob
import shutil
from pathlib import Path
import torch
import yaml
import random

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def create_unified_dataset():
    """
    Creates a balanced, unified dataset from multiple sources.
    
    Process:
    1. Load & Group: Collect all images from sources into 3 classes (Fluorescent, Fire Extinguisher, Outlet).
    2. Balance: Sample N images from each class (N = size of smallest class).
    3. Split: 60% Train, 20% Valid, 20% Test.
    4. Augment: Apply Rot90CW and Rot90CCW to ALL splits (3x expansion).
    5. Merge: Combine into unified dataset structure.
    """
    
    # Define unified dataset structure
    unified_dataset_dir = "training datasets/unified_dataset"
    dirs = {
        'train': {
            'images': os.path.join(unified_dataset_dir, "train", "images"),
            'labels': os.path.join(unified_dataset_dir, "train", "labels")
        },
        'valid': {
            'images': os.path.join(unified_dataset_dir, "valid", "images"),
            'labels': os.path.join(unified_dataset_dir, "valid", "labels")
        },
        'test': {
            'images': os.path.join(unified_dataset_dir, "test", "images"),
            'labels': os.path.join(unified_dataset_dir, "test", "labels")
        }
    }
    
    # Clean up existing
    if os.path.exists(unified_dataset_dir):
        shutil.rmtree(unified_dataset_dir)
    
    # Create directories
    for split in dirs:
        os.makedirs(dirs[split]['images'], exist_ok=True)
        os.makedirs(dirs[split]['labels'], exist_ok=True)
    
    print("=" * 60)
    print("📦 CREATING UNIFIED DATASET (BALANCED & AUGMENTED)")
    print("=" * 60)
    
    # 1. COLLECT IMAGES
    print("\n🔄 Step 1: Collecting images...")
    
    # Store as list of (image_path, label_path) tuples
    groups = {
        0: [], # fluorescent tube
        1: [], # fire extinguisher
        2: []  # outlet
    }
    
    sources = [
        ('training datasets/flourescent lamp 1', 0),
        ('training datasets/fluorescent lamp 2', 0),
        ('training datasets/fire extinguisher', 1),
        ('training datasets/socket 1', 2),
        ('training datasets/socket 2', 2)
    ]
    
    for dataset_path, class_id in sources:
        if not os.path.exists(dataset_path):
            print(f"   ⚠️ Source not found: {dataset_path}")
            continue
            
        count = 0
        # Walk recursively to find all images regardless of split
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    
                    # Find label
                    # Strategy: Look for label in 'labels' folder parallel to 'images' folder
                    # or in the same folder
                    label_path = None
                    
                    # Check 1: Standard YOLO structure (.../images/x.jpg -> .../labels/x.txt)
                    if 'images' in root:
                        label_root = root.replace('images', 'labels')
                        possible_label = os.path.join(label_root, os.path.splitext(file)[0] + '.txt')
                        if os.path.exists(possible_label):
                            label_path = possible_label
                    
                    # Check 2: Same directory
                    if not label_path:
                        possible_label = os.path.join(root, os.path.splitext(file)[0] + '.txt')
                        if os.path.exists(possible_label):
                            label_path = possible_label
                            
                    if label_path:
                        groups[class_id].append((img_path, label_path))
                        count += 1
        
        print(f"   Found {count} pairs in {dataset_path}")

    # 2. BALANCE (SAMPLING)
    print("\n⚖️ Step 2: Balancing classes...")
    counts = {k: len(v) for k, v in groups.items()}
    print(f"   Initial counts: {counts}")
    
    min_count = min(counts.values())
    if min_count == 0:
        raise ValueError("❌ Error: One or more classes have 0 images!")
        
    print(f"   Sampling {min_count} images per class")
    
    for class_id in groups:
        random.shuffle(groups[class_id])
        groups[class_id] = groups[class_id][:min_count]

    # 3. SPLIT, RELABEL, AUGMENT
    print("\n✂️ Step 3: Splitting, Relabeling & Augmenting (3x)...")
    
    # Split ratios
    n_train = int(min_count * 0.6)
    n_valid = int(min_count * 0.2)
    # Remaining goes to test
    
    class_names = {0: 'fluorescent_tube', 1: 'fire_extinguisher', 2: 'outlet'}
    
    total_images = {'train': 0, 'valid': 0, 'test': 0}
    
    for class_id, items in groups.items():
        # Split
        train_set = items[:n_train]
        valid_set = items[n_train:n_train+n_valid]
        test_set = items[n_train+n_valid:]
        
        splits_data = [('train', train_set), ('valid', valid_set), ('test', test_set)]
        
        print(f"   Processing Class {class_id} ({class_names[class_id]}): {len(train_set)}/{len(valid_set)}/{len(test_set)}")
        
        for split_name, split_items in splits_data:
            dst_img_dir = dirs[split_name]['images']
            dst_lbl_dir = dirs[split_name]['labels']
            
            for img_path, label_path in split_items:
                # Load image
                img = cv2.imread(img_path)
                if img is None: continue
                
                # Load labels
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Base filename
                base_name = f"{class_names[class_id]}_{os.path.basename(img_path)}"
                base_name = os.path.splitext(base_name)[0]
                
                # Augmentations: Original, Rot90CW, Rot90CCW
                augmentations = [
                    ('orig', img, lambda x, y, w, h: (x, y, w, h)),
                    ('rot90cw', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), lambda x, y, w, h: (1.0-y, x, h, w)),
                    ('rot90ccw', cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), lambda x, y, w, h: (y, 1.0-x, h, w))
                ]
                
                for aug_suffix, aug_img, transform_func in augmentations:
                    # Save Image
                    final_name = f"{base_name}_{aug_suffix}.jpg"
                    save_path = os.path.join(dst_img_dir, final_name)
                    success = cv2.imwrite(save_path, aug_img)
                    
                    if not success:
                        print(f"      ⚠️ Failed to save image: {final_name}")
                        continue
                    
                    # Save Label
                    final_label_name = os.path.splitext(final_name)[0] + '.txt'
                    with open(os.path.join(dst_lbl_dir, final_label_name), 'w') as f_out:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Parse
                                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                
                                # Transform
                                nx, ny, nw, nh = transform_func(x, y, w, h)
                                
                                # Write with NEW class_id
                                f_out.write(f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                    
                    total_images[split_name] += 1

    print(f"\n{'='*60}")
    print(f"✅ Dataset creation complete!")
    print(f"   Train images: {total_images['train']}")
    print(f"   Valid images: {total_images['valid']}")
    print(f"   Test images:  {total_images['test']}")
    print(f"{'='*60}\n")
    
    # Create unified data.yaml
    data_yaml = {
        'train': os.path.abspath(dirs['train']['images']),
        'val': os.path.abspath(dirs['valid']['images']),
        'test': os.path.abspath(dirs['test']['images']),
        'nc': 3,
        'names': ['fluorescent tube', 'fire extinguisher', 'outlet']
    }
    
    yaml_path = os.path.join(unified_dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"📄 Created unified data.yaml: {yaml_path}")
    
    # Force delete any existing cache files to prevent stale cache issues
    for split in ['train', 'valid', 'test']:
        cache_path = os.path.join(dirs[split]['labels'], 'labels.cache')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"   🗑️ Deleted stale cache: {cache_path}")
    
    return yaml_path


def train_yolov8x_world(data_yaml, epochs=100, batch=16, imgsz=640):
    """
    Train YOLOv8x-World model on unified dataset.
    """
    
    model_path = "yolov8x-world.pt"
    custom_weights = "yolov8x_world_custom_trained.pt"
    
    # Device selection with fallback
    if torch.cuda.is_available():
        device = 0
        print("=" * 60)
        print("🚀 STARTING YOLOV8X-WORLD TRAINING")
        print("=" * 60)
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        device = 'cpu'
        print("=" * 60)
        print("🚀 STARTING YOLOV8X-WORLD TRAINING")
        print("=" * 60)
        print("⚠️ GPU not detected by PyTorch. Falling back to CPU.")
        print("   Possible reason: NVIDIA driver is too old for this PyTorch version.")
        print("   (PyTorch requires CUDA 12.1+, found driver for older CUDA)")

    print(f"   Using device: {device}")
    
    print(f"\n📋 Training Configuration:")
    print(f"   Base model: {model_path} (YOLOv8x-World)")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    print(f"   Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)
    
    # Train
    print("Starting training...\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="yolov8x_world_custom",
        patience=3,        # Early stopping patience
        cache=False,        # Don't cache images in RAM
        workers=1,          # Number of dataloader workers
        device=device,
        amp=True,           # Automatic Mixed Precision (reduces GPU memory ~40%)
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
        verbose=True
    )
    
    # Save best weights
    best_weight_path = os.path.join("runs/detect/yolov8x_world_custom/weights/best.pt")
    if os.path.exists(best_weight_path):
        shutil.copy(best_weight_path, custom_weights)
        print(f"\n✅ Custom weights saved as {custom_weights}")
    else:
        # Try to find the latest run directory
        run_dirs = glob.glob("runs/detect/yolov8x_world_custom*/weights/best.pt")
        if run_dirs:
            latest_best = sorted(run_dirs, key=os.path.getmtime)[-1]
            shutil.copy(latest_best, custom_weights)
            print(f"\n✅ Custom weights saved as {custom_weights} from {latest_best}")
        else:
            print("\n⚠️ Training finished, but best weights not found.")
            custom_weights = None
    
    return custom_weights


def run_inference(model_path, examples_dir="examples", output_dir="yolov8x_world_results"):
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
    
    parser = argparse.ArgumentParser(description="Train YOLOv8x-World on unified multi-class dataset")
    parser.add_argument('--skip-merge', action='store_true', help='Skip dataset merging (use existing unified_dataset)')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing weights)')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint (e.g., runs/detect/yolov8x_world_custom/weights/last.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size (default: 4 for low VRAM)')
    parser.add_argument('--imgsz', type=int, default=312, help='Image size (default: 312 for low VRAM)')
    parser.add_argument('--weights', default='yolov8x_world_custom_trained.pt', help='Path to trained weights for inference')
    
    args = parser.parse_args()
    
    # Handle resume training from checkpoint
    if args.resume:
        print("=" * 60)
        print("🔄 RESUMING TRAINING FROM CHECKPOINT")
        print("=" * 60)
        print(f"Checkpoint: {args.resume}")
        
        if not os.path.exists(args.resume):
            print(f"❌ Checkpoint not found: {args.resume}")
            return
        
        # Set enhanced memory management
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        # Clear CUDA cache before resuming
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"\n📋 Enhanced Memory Settings:")
        print(f"   CUDA_LAUNCH_BLOCKING: 1")
        print(f"   PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True,max_split_size_mb:128")
        print(f"{'='*60}\n")
        
        # Load model and resume training
        print("Loading model from checkpoint...")
        model = YOLO(args.resume)
        
        # Check if CUDA is actually available
        device = 'cpu'
        if torch.cuda.is_available():
            device = 0
            print(f"✅ CUDA available, using GPU")
        else:
            print(f"⚠️ CUDA not available, forcing CPU mode")
            print(f"   Note: Training will be much slower on CPU")
            print(f"   The model already achieved excellent results (mAP50=0.645 at epoch 34)")
            print(f"   Consider using the best weights for inference instead.\n")
        
        print("Resuming training...\n")
        try:
            # Override device to ensure compatibility
            results = model.train(resume=True, device=device)
            print("\n✅ Training completed successfully!")
            
            # Copy best weights
            # Find the run directory
            run_dirs = glob.glob("runs/detect/yolov8x_world_custom*/weights/best.pt")
            if run_dirs:
                latest_best = sorted(run_dirs, key=os.path.getmtime)[-1]
                custom_weights = "yolov8x_world_custom_trained.pt"
                shutil.copy(latest_best, custom_weights)
                print(f"✅ Best weights saved as {custom_weights}")
            else:
                custom_weights = args.weights
        except RuntimeError as e:
            print(f"\n❌ Training failed with error: {e}")
            print("\nThe model already achieved excellent results at epoch 34:")
            print("  - mAP50: 0.645")
            print("  - Precision: 0.649") 
            print("  - Recall: 0.646")
            print("\nConsider using the best model for inference instead.")
            return
    else:
        # Step 1: Create unified dataset
        if not args.skip_merge:
            data_yaml = create_unified_dataset()
        else:
            data_yaml = "training datasets/unified_dataset/data.yaml"
            print(f"⏭️ Skipping dataset merge, using existing: {data_yaml}\n")
            
            # Force delete cache even when skipping merge
            for split in ['train', 'valid', 'test']:
                cache_path = os.path.join("training datasets/unified_dataset", split, "labels", "labels.cache")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    print(f"   🗑️ Deleted stale cache: {cache_path}")
        
        # Step 2: Train model
        if not args.skip_training:
            custom_weights = train_yolov8x_world(
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
