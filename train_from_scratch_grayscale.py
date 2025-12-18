"""
Train YOLOv11x model COMPLETELY FROM SCRATCH on augmented color images.
No pretrained weights - builds the model architecture from scratch.

This script:
1. Augments all training images (Rot90 CW/CCW)
2. Creates balanced dataset with 80/20 train/valid split
3. Trains YOLOv11x from scratch (no transfer learning)
4. Evaluates on test set
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
import numpy as np

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def create_augmented_dataset():
    """
    Create a unified augmented dataset with balanced classes.
    All images augmented (Rot90) before training.
    Split: 80% train, 20% validation
    """
    
    # Define unified dataset structure
    dataset_dir = "training datasets/augmented_from_scratch"
    train_images_dir = os.path.join(dataset_dir, "train", "images")
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    val_images_dir = os.path.join(dataset_dir, "valid", "images")
    val_labels_dir = os.path.join(dataset_dir, "valid", "labels")
    
    # Create directories
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("=" * 70)
    print("🎨 CREATING AUGMENTED DATASET FOR TRAINING FROM SCRATCH")
    print("=" * 70)
    print("Strategy: Augment (Rot90), Balance, 80/20 split")
    print("=" * 70)
    
    # Define source datasets
    datasets_info = [
        {
            'name': 'Fluorescent Lamp 1',
            'path': 'training datasets/flourescent lamp 1',
            'class_id': 0,
            'class_name': 'fluorescent tube'
        },
        {
            'name': 'Fluorescent Lamp 2',
            'path': 'training datasets/fluorescent lamp 2',
            'class_id': 0,
            'class_name': 'fluorescent tube'
        },
        {
            'name': 'Fluorescent Lamp Test',
            'path': 'training datasets/fluor_lamp_test/Fluorescent light detection.v1-original-with-augmentation.yolov8',
            'class_id': 0,
            'class_name': 'fluorescent tube'
        },
        {
            'name': 'Fire Extinguisher',
            'path': 'training datasets/fire extinguisher',
            'class_id': 1,
            'class_name': 'fire extinguisher'
        },
        {
            'name': 'Fire Extinguisher Test',
            'path': 'training datasets/fire_extinguisher_test/Fire Extinguisher Finder.v1i.yolov8',
            'class_id': 1,
            'class_name': 'fire extinguisher'
        },
        {
            'name': 'Socket 1',
            'path': 'training datasets/socket 1',
            'class_id': 2,
            'class_name': 'outlet'
        },
        {
            'name': 'Socket 2',
            'path': 'training datasets/socket 2',
            'class_id': 2,
            'class_name': 'outlet'
        },
        {
            'name': 'Socket Test Close',
            'path': 'training datasets/socket_test_close/Ai project.v1i.yolov8',
            'class_id': 2,
            'class_name': 'outlet'
        }
    ]
    
    # Step 1: Collect all images by class
    print("\n📥 STEP 1: Collecting images from all datasets...")
    class_image_data = {0: [], 1: [], 2: []}
    
    for dataset_info in datasets_info:
        dataset_path = dataset_info['path']
        class_id = dataset_info['class_id']
        dataset_name = dataset_info['name']
        
        print(f"\n📂 Scanning: {dataset_name}")
        
        if not os.path.exists(dataset_path):
            print(f"   ⚠️ Dataset not found: {dataset_path}")
            continue
        
        # Look for images in multiple possible locations
        possible_locations = [
            (os.path.join(dataset_path, 'train', 'images'), os.path.join(dataset_path, 'train', 'labels')),
            (os.path.join(dataset_path, 'valid', 'images'), os.path.join(dataset_path, 'valid', 'labels')),
            (os.path.join(dataset_path, 'images'), os.path.join(dataset_path, 'labels'))
        ]
        
        dataset_image_count = 0
        for src_images, src_labels in possible_locations:
            if not os.path.exists(src_images):
                continue
            
            # Get all image files
            image_files = glob.glob(os.path.join(src_images, '*.jpg')) + \
                         glob.glob(os.path.join(src_images, '*.jpeg')) + \
                         glob.glob(os.path.join(src_images, '*.png'))
            
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(src_labels, label_name)
                
                if os.path.exists(label_path):
                    class_image_data[class_id].append((img_path, label_path, class_id, dataset_info['class_name']))
                    dataset_image_count += 1
        
        print(f"   ✅ Collected {dataset_image_count} images")
    
    # Print class distribution
    print(f"\n{'='*70}")
    print(f"📊 CLASS DISTRIBUTION (Before Balancing):")
    print(f"{'='*70}")
    class_names = {0: 'fluorescent tube', 1: 'fire extinguisher', 2: 'outlet'}
    for class_id in [0, 1, 2]:
        print(f"   {class_names[class_id]}: {len(class_image_data[class_id])} images")
    
    # Step 2: Balance classes
    min_class_size = min(len(class_image_data[0]), len(class_image_data[1]), len(class_image_data[2]))
    
    if min_class_size == 0:
        print("\n❌ One or more classes have no images! Exiting.")
        return None
    
    print(f"\n{'='*70}")
    print(f"⚖️ BALANCING CLASSES:")
    print(f"{'='*70}")
    print(f"   Smallest class has {min_class_size} images")
    print(f"   Downsampling all classes to match...\n")
    
    random.seed(42)  # For reproducibility
    all_image_data = []
    
    for class_id in [0, 1, 2]:
        class_images = class_image_data[class_id]
        if len(class_images) > min_class_size:
            sampled_images = random.sample(class_images, min_class_size)
            print(f"   {class_names[class_id]}: Sampled {min_class_size} from {len(class_images)}")
            all_image_data.extend(sampled_images)
        else:
            print(f"   {class_names[class_id]}: Using all {len(class_images)} images")
            all_image_data.extend(class_images)
    
    total_images = len(all_image_data)
    print(f"\n{'='*70}")
    print(f"✅ Balanced dataset: {total_images} images ({min_class_size} per class × 3)")
    print(f"{'='*70}")
    
    if total_images == 0:
        print("❌ No images found! Exiting.")
        return None
    
    # Step 3: Shuffle and split 80/20
    print("\n🔀 STEP 2: Shuffling and splitting (80% train, 20% valid)...")
    random.shuffle(all_image_data)
    
    train_split_idx = int(total_images * 0.8)
    train_data = all_image_data[:train_split_idx]
    valid_data = all_image_data[train_split_idx:]
    
    print(f"   📊 Train: {len(train_data)} images (80%)")
    print(f"   📊 Valid: {len(valid_data)} images (20%)")
    
    # Verify class distribution in splits
    print(f"\n{'='*70}")
    print(f"📊 CLASS DISTRIBUTION PER SPLIT:")
    print(f"{'='*70}")
    for split_name, split_data in [('Train', train_data), ('Valid', valid_data)]:
        class_counts = {0: 0, 1: 0, 2: 0}
        for _, _, class_id, _ in split_data:
            class_counts[class_id] += 1
        print(f"   {split_name}:")
        for class_id in [0, 1, 2]:
            print(f"      {class_names[class_id]}: {class_counts[class_id]} images")
    print(f"{'='*70}")
    
    # Step 4: Augment and save
    print(f"\n🎨 STEP 3: Augmenting and saving...")
    print(f"   All images will be augmented (Rot90 CW/CCW)")
    
    splits_info = [
        ('train', train_data, train_images_dir, train_labels_dir),
        ('valid', valid_data, val_images_dir, val_labels_dir)
    ]
    
    for split_name, split_data, dst_images, dst_labels in splits_info:
        print(f"\n   Processing {split_name} split...")
        processed = 0
        grayscale_converted = 0
        
        for img_path, label_path, class_id, class_name in split_data:
            img_name = os.path.basename(img_path)
            unique_name = f"{class_name.replace(' ', '_')}_{Path(img_path).parent.parent.name.replace(' ', '_')}_{img_name}"
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"      ⚠️ Failed to load {img_path}, skipping...")
                continue
            
            # Use original color image
            # Define augmentations: (suffix, image, coordinate_transform_func)
            # Original: x, y, w, h
            # Rot 90 CW: new_x = 1-y, new_y = x, new_w = h, new_h = w
            # Rot 90 CCW: new_x = y, new_y = 1-x, new_w = h, new_h = w
            
            augmentations = [
                ('orig', img, lambda x, y, w, h: (x, y, w, h)),
                ('rot90cw', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), lambda x, y, w, h: (1.0-y, x, h, w)),
                ('rot90ccw', cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), lambda x, y, w, h: (y, 1.0-x, h, w))
            ]
            
            base_filename = os.path.splitext(unique_name)[0]
            
            for aug_suffix, aug_img, transform_func in augmentations:
                # Save image
                aug_filename = f"{base_filename}_{aug_suffix}.jpg"
                dst_img_path = os.path.join(dst_images, aug_filename)
                
                success = cv2.imwrite(dst_img_path, aug_img)
                if not success:
                    print(f"      ⚠️ Failed to save {dst_img_path}")
                    continue
                
                if aug_suffix == 'orig':
                    grayscale_converted += 1
                
                # Process labels
                dst_label_path = os.path.join(dst_labels, os.path.splitext(aug_filename)[0] + '.txt')
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                with open(dst_label_path, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Parse original coordinates
                            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            
                            # Transform coordinates
                            nx, ny, nw, nh = transform_func(x, y, w, h)
                            
                            # Write new label with correct class ID
                            f.write(f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
            
            processed += 1
        
        print(f"   ✅ {split_name}: {processed} images processed, {grayscale_converted} converted to B&W")
    
    print(f"\n{'='*70}")
    print(f"✅ AUGMENTED DATASET CREATED!")
    print(f"   Train: {len(train_data)} images (80%)")
    print(f"   Valid: {len(valid_data)} images (20%)")
    print(f"   Total: {total_images} images")
    print(f"   🎨 Color images (Augmented)")
    print(f"   Classes: {list(class_names.values())}")
    print(f"{'='*70}\n")
    
    # Create data.yaml
    data_yaml = {
        'train': os.path.abspath(train_images_dir),
        'val': os.path.abspath(val_images_dir),
        'nc': 3,
        'names': ['fluorescent tube', 'fire extinguisher', 'outlet']
    }
    
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"📄 Created data.yaml: {yaml_path}")
    print(f"   Classes: {data_yaml['names']}\n")
    
    return yaml_path


def train_from_scratch(data_yaml, epochs=200, batch=8, imgsz=640, model_size='x'):
    """
    Train YOLOv11 model FROM SCRATCH (no pretrained weights).
    This will take significantly longer but learns specifically from your grayscale data.
    
    Args:
        model_size: 'n', 's', 'm', 'l', or 'x' (nano to extra-large)
                   For lower GPU load, use 'n' (nano) or 's' (small)
    """
    
    # Use cfg file to build model from scratch
    model_cfg = f"yolo11{model_size}.yaml"  # Architecture configuration
    custom_weights = f"yolov11{model_size}_augmented_from_scratch.pt"
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print(f"🚀 TRAINING YOLOV11{model_size.upper()} FROM SCRATCH (NO PRETRAINED WEIGHTS)")
    print("=" * 70)
    print("⚠️ NOTE: Training from scratch takes MUCH longer than transfer learning")
    print("   Expected time: 10-20x longer than using pretrained weights")
    print("   Recommended: At least 200+ epochs for good results")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️ WARNING: No GPU detected!")
        print("   Training from scratch on CPU is NOT recommended")
        print("   This could take days or weeks to complete")
        print("   Consider using Google Colab or a GPU instance")
    
    print(f"\n📋 Training Configuration:")
    print(f"   Model: YOLOv11{model_size} (built from scratch)")
    print(f"   Model size: {model_size.upper()} - {'NANO (lowest GPU)' if model_size == 'n' else 'SMALL (low GPU)' if model_size == 's' else 'MEDIUM' if model_size == 'm' else 'LARGE' if model_size == 'l' else 'EXTRA-LARGE (highest GPU)'}")
    print(f"   Pretrained weights: NONE (learning from random initialization)")
    print(f"   Dataset: {data_yaml}")
    print(f"   Image type: Color (3-channel)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    print(f"   Device: {device}")
    print(f"   Learning rate: 0.01 (higher for scratch training)")
    print(f"={'='*70}\n")
    
    # Build model from scratch using config file
    print(f"Building YOLOv11x architecture from scratch...")
    print(f"Using configuration: {model_cfg}")
    
    # Create model from config (no pretrained weights)
    model = YOLO(model_cfg)
    
    print(f"✅ Model initialized with random weights")
    print(f"   Total parameters will be optimized from scratch\n")
    
    # Train
    print("🎯 Starting training from scratch...\n")
    print("=" * 70)
    print("TRAINING TIPS FOR FROM-SCRATCH MODELS:")
    print("  - First 50 epochs: Model learns basic features (edges, shapes)")
    print("  - 50-100 epochs: Learns object-specific features")
    print("  - 100-200 epochs: Fine-tunes detection accuracy")
    print("  - Monitor validation loss - should steadily decrease")
    print("  - Don't stop too early! Model needs time to converge")
    print("=" * 70)
    print()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=f"yolov11{model_size}_augmented_scratch",
        patience=50,        # Higher patience for scratch training
        cache=False,        # Don't cache images (saves GPU RAM)
        workers=1,          # Reduce workers to save RAM
        device=device,
        amp=True,           # Mixed precision training (saves GPU RAM)
        
        # Learning rate schedule optimized for scratch training
        lr0=0.01,           # Higher initial LR for scratch training
        lrf=0.001,          # Lower final LR
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,    # Longer warmup for scratch training
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Memory optimization
        close_mosaic=10,    # Disable mosaic augmentation in last 10 epochs (saves memory)
        
        # Other settings
        plots=True,
        save=True,
        save_period=20,     # Save checkpoint every 20 epochs
        val=True,
        verbose=True,
        exist_ok=True
    )
    
    # Save best weights
    best_weight_path = os.path.join(f"runs/detect/yolov11{model_size}_grayscale_scratch/weights/best.pt")
    if os.path.exists(best_weight_path):
        shutil.copy(best_weight_path, custom_weights)
        print(f"\n✅ Best weights saved as: {custom_weights}")
    else:
        # Try to find latest run
        run_dirs = glob.glob(f"runs/detect/yolov11{model_size}_grayscale_scratch*/weights/best.pt")
        if run_dirs:
            latest_best = sorted(run_dirs, key=os.path.getmtime)[-1]
            shutil.copy(latest_best, custom_weights)
            print(f"\n✅ Best weights saved as: {custom_weights}")
        else:
            print("\n⚠️ Training finished but best weights not found")
            custom_weights = None
    
    return custom_weights


def evaluate_model(model_path, data_yaml):
    """
    Evaluate the trained model on validation set.
    """
    print("\n" + "=" * 70)
    print("🧪 EVALUATING MODEL ON VALIDATION SET")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Running evaluation...\n")
    results = model.val(
        data=data_yaml,
        device=device,
        batch=1,
        imgsz=640,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print(f"✅ EVALUATION RESULTS:")
    print(f"   mAP50-95: {results.box.map:.3f}")
    print(f"   mAP50: {results.box.map50:.3f}")
    print(f"   Precision: {results.box.mp:.3f}")
    print(f"   Recall: {results.box.mr:.3f}")
    print(f"{'='*70}\n")
    
    return results


def run_inference(model_path, examples_dir="examples", output_dir="augmented_scratch_results"):
    """
    Run inference on example images.
    """
    print("\n" + "=" * 70)
    print("🔍 RUNNING INFERENCE ON EXAMPLES")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
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
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️ Could not load {img_path}")
            continue
        
        # Run detection on color image
        results = model.predict(source=img, conf=0.25, imgsz=640, device=device, verbose=False)
        
        # Draw detections
        num_detections = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Draw on image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
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
    
    print(f"\n{'='*70}")
    print(f"✅ Inference complete!")
    print(f"   Processed {len(image_files)} images")
    print(f"   Total detections: {total_detections}")
    print(f"   Results saved to: {output_dir}/")
    print(f"{'='*70}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train YOLOv11x from scratch on augmented color images"
    )
    parser.add_argument('--skip-dataset', action='store_true', 
                       help='Skip dataset creation (use existing)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing weights)')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 80)')
    parser.add_argument('--batch', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--imgsz', type=int, default=512,
                       help='Image size (default: 512)')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge). Use "n" or "s" for low GPU RAM (default: n)')
    parser.add_argument('--weights', default='yolov11n_augmented_from_scratch.pt',
                       help='Path to model weights')
    
    args = parser.parse_args()
    
    data_yaml = "training datasets/augmented_from_scratch/data.yaml"
    
    # Evaluate only mode
    if args.evaluate_only:
        if not os.path.exists(args.weights):
            print(f"❌ Weights not found: {args.weights}")
            return
        if not os.path.exists(data_yaml):
            print(f"❌ Dataset config not found: {data_yaml}")
            return
        evaluate_model(args.weights, data_yaml)
        run_inference(args.weights)
        return
    
    # Step 1: Create augmented dataset
    if not args.skip_dataset:
        data_yaml = create_augmented_dataset()
        if not data_yaml:
            print("❌ Dataset creation failed!")
            return
    else:
        if not os.path.exists(data_yaml):
            print(f"❌ Dataset config not found: {data_yaml}")
            print("   Run without --skip-dataset first")
            return
        print(f"⏭️ Skipping dataset creation, using: {data_yaml}\n")
    
    # Step 2: Train from scratch
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("⚠️ IMPORTANT: Training from scratch requires:")
        print("   - Good GPU (4GB+ VRAM for nano/small, 8GB+ for medium/large/xlarge)")
        print("   - Patience (training will take much longer)")
        print("   - At least 200+ epochs for convergence")
        print("=" * 70)
        print(f"\n💡 SELECTED MODEL SIZE: {args.model_size.upper()}")
        print(f"   GPU RAM requirements:")
        print(f"     nano (n):   ~2-4 GB VRAM  ✅ Best for low-end GPUs")
        print(f"     small (s):  ~4-6 GB VRAM  ✅ Good balance")
        print(f"     medium (m): ~6-8 GB VRAM")
        print(f"     large (l):  ~8-12 GB VRAM")
        print(f"     xlarge (x): ~12-16 GB VRAM ⚠️ Requires high-end GPU")
        print("=" * 70)
        
        response = input("\nContinue with training from scratch? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Training cancelled.")
            return
        
        model_weights = train_from_scratch(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            model_size=args.model_size
        )
    else:
        model_weights = args.weights
        print(f"⏭️ Skipping training, using: {model_weights}\n")
    
    # Step 3: Evaluate and run inference
    if model_weights and os.path.exists(model_weights):
        evaluate_model(model_weights, data_yaml)
        run_inference(model_weights)
    else:
        print("\n⚠️ No model weights available for evaluation")


if __name__ == '__main__':
    main()
