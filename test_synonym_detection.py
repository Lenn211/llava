"""
Test synonym detection capabilities of YOLOv8-World models.
This script tests whether the custom model can detect objects using different class names.
"""

from ultralytics import YOLO
import os
import glob

def test_model_with_synonyms(model_path, image_folder, class_variants):
    """Test a model with different class name variants"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Get test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_paths:
        print(f"❌ No images found in {image_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_path}")
    print(f"Images: {len(image_paths)} found in {image_folder}")
    print(f"{'='*70}")
    
    model = YOLO(model_path)
    
    # Get original class names
    print(f"\n📝 Original trained classes: {list(model.names.values())}")
    
    # Test each set of class variants
    for category, variants in class_variants.items():
        print(f"\n{'─'*70}")
        print(f"Testing category: {category.upper()}")
        print(f"Class variants: {variants}")
        print(f"{'─'*70}")
        
        for variant in variants:
            # Set the single class variant
            model.set_classes([variant])
            
            # Run detection on first image
            test_image = image_paths[0]
            results = model.predict(source=test_image, conf=0.25, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    class_name = result.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    detections.append((class_name, confidence))
            
            if detections:
                print(f"  ✅ '{variant}' → Detected {len(detections)} object(s):")
                for class_name, conf in detections:
                    print(f"      - {class_name} (conf: {conf:.2f})")
            else:
                print(f"  ❌ '{variant}' → No detections")
    
    print(f"\n{'='*70}")
    print("Testing with ALL variants at once...")
    print(f"{'='*70}")
    
    # Test with all variants together
    all_variants = []
    for variants in class_variants.values():
        all_variants.extend(variants)
    
    model.set_classes(all_variants)
    print(f"Set classes: {all_variants}")
    
    # Run detection on first image
    test_image = image_paths[0]
    results = model.predict(source=test_image, conf=0.25, verbose=False)
    
    detections = []
    if results and len(results) > 0:
        result = results[0]
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append((class_name, confidence))
    
    if detections:
        print(f"\n✅ All variants → Detected {len(detections)} object(s):")
        for class_name, conf in detections:
            print(f"   - {class_name} (conf: {conf:.2f})")
    else:
        print(f"\n❌ All variants → No detections")
    
    print()


if __name__ == "__main__":
    print("\n🧪 YOLOV8-WORLD SYNONYM DETECTION TEST")
    print("="*70)
    
    # Define class variants to test
    class_variants = {
        "sockets": [
            "outlet",                # Original trained class
            "power outlet",          # Synonym
            "wall socket",           # Synonym
            "electrical socket",     # Synonym
            "power point",           # Synonym
        ],
        "fire_safety": [
            "fire extinguisher",     # Original trained class
            "fire safety equipment", # Synonym
            "fire suppression device", # Synonym
        ],
        "lighting": [
            "fluorescent tube",      # Original trained class
            "light fixture",         # Synonym
            "ceiling light",         # Synonym
            "lamp",                  # Synonym
        ]
    }
    
    # Test both models
    test_image_folder = "gemini_test_folder"
    
    print("\n" + "="*70)
    print("TEST 1: YOLOv8x-World (Base Model)")
    print("="*70)
    test_model_with_synonyms("yolov8x-world.pt", test_image_folder, class_variants)
    
    print("\n" + "="*70)
    print("TEST 2: Custom YOLOv8x-World Model")
    print("="*70)
    test_model_with_synonyms("custom_yolov8x.pt", test_image_folder, class_variants)
    
    print("\n💡 INTERPRETATION:")
    print("   - If original classes (outlet, fire extinguisher, fluorescent tube) detect:")
    print("     → Model is working correctly")
    print("   - If synonyms also detect:")
    print("     → Open-vocabulary detection is working!")
    print("   - If only original classes detect:")
    print("     → Model needs exact class names or better semantic matching")
    print()
