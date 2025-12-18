"""
Test if the custom model can detect objects WITHOUT using set_classes().
This will use the model's default trained classes.
"""

from ultralytics import YOLO
import os
import glob

def test_model_default_detection(model_path, image_folder):
    """Test model with its default trained classes (no set_classes)"""
    
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
    print(f"{'='*70}")
    
    model = YOLO(model_path)
    
    print(f"📝 Trained classes: {list(model.names.values())}")
    print(f"\n🔍 Running detection on {len(image_paths)} images...")
    print(f"   (Using DEFAULT trained classes, NO set_classes() call)")
    print(f"{'─'*70}\n")
    
    total_detections = 0
    
    for idx, image_path in enumerate(image_paths, 1):
        results = model.predict(source=image_path, conf=0.25, verbose=False)
        
        image_detections = []
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                class_name = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                image_detections.append((class_name, confidence))
                total_detections += 1
        
        if image_detections:
            print(f"  Image {idx}: {os.path.basename(image_path)}")
            for class_name, conf in image_detections:
                print(f"    ✅ {class_name} (confidence: {conf:.2f})")
        else:
            print(f"  Image {idx}: {os.path.basename(image_path)}")
            print(f"    ❌ No detections")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_detections} total detections across {len(image_paths)} images")
    print(f"{'='*70}\n")
    
    return total_detections


if __name__ == "__main__":
    print("\n🧪 YOLOV8-WORLD DEFAULT DETECTION TEST")
    print("Testing models WITHOUT using set_classes()")
    print("="*70)
    
    test_image_folder = "gemini_test_folder"
    
    print("\n" + "─"*70)
    print("TEST 1: Base YOLOv8x-World Model")
    print("─"*70)
    base_detections = test_model_default_detection("yolov8x-world.pt", test_image_folder)
    
    print("\n" + "─"*70)
    print("TEST 2: Custom YOLOv8x-World Model")
    print("─"*70)
    custom_detections = test_model_default_detection("custom_yolov8x.pt", test_image_folder)
    
    print("\n💡 INTERPRETATION:")
    if custom_detections > 0:
        print("   ✅ Custom model IS detecting objects with default classes")
        print("   ⚠️  But it may NOT work with set_classes() for synonyms")
        print("   → This is a limitation of how the model was trained")
    else:
        print("   ❌ Custom model is NOT detecting ANY objects")
        print("   → Possible issues:")
        print("     1. Test images don't contain trained objects")
        print("     2. Confidence threshold is too high (try conf=0.10)")
        print("     3. Model training had issues")
    print()
