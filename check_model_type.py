"""
Check if a YOLO model is a YOLOv8-World model or standard YOLOv8.
This determines whether it can use open-vocabulary detection with synonyms.
"""

from ultralytics import YOLO
import os

def check_model_type(model_path):
    """Check if a model is YOLOv8-World or standard YOLOv8"""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Checking: {model_path}")
    print(f"{'='*60}")
    
    try:
        model = YOLO(model_path)
        
        # Check model architecture
        model_type = type(model.model).__name__
        print(f"\n📊 Model Type: {model_type}")
        
        # Check if it's a World model
        is_world = "world" in model_path.lower() or "World" in model_type
        
        if is_world:
            print("✅ This IS a YOLOv8-World model")
            print("   → Can use open-vocabulary detection")
            print("   → Can detect synonyms with set_classes()")
        else:
            print("❌ This is NOT a YOLOv8-World model")
            print("   → Cannot use open-vocabulary detection")
            print("   → Only detects trained class names exactly")
        
        # Get class names
        if hasattr(model, 'names'):
            print(f"\n📝 Trained Classes ({len(model.names)}):")
            for idx, name in model.names.items():
                print(f"   {idx}: {name}")
        
        # Model info
        print(f"\n🔧 Model Info:")
        print(f"   Task: {model.task}")
        if hasattr(model.model, 'yaml'):
            print(f"   Architecture: {model.model.yaml.get('backbone', 'N/A')}")
        
        return is_world
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 YOLO MODEL TYPE CHECKER")
    print("="*60)
    
    # Check both models
    models_to_check = [
        "yolov8x-world.pt",
        "custom_yolov8x.pt"
    ]
    
    results = {}
    for model_path in models_to_check:
        is_world = check_model_type(model_path)
        results[model_path] = is_world
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_path, is_world in results.items():
        status = "✅ YOLOv8-World" if is_world else "❌ Standard YOLOv8"
        print(f"{model_path}: {status}")
    
    print("\n💡 RECOMMENDATION:")
    if not results.get("custom_yolov8x.pt", False):
        print("   Your custom model is NOT a YOLOv8-World model.")
        print("   To use synonym detection, you need to:")
        print("   1. Train with: python train_yolov8x_world_unified.py")
        print("   2. This will create a YOLOv8x-WORLD model")
        print("   3. Then it can detect all synonyms!")
    else:
        print("   Your custom model IS a YOLOv8-World model.")
        print("   It should already support synonym detection!")
        print("   Make sure to use model.set_classes() with your synonyms.")
    
    print()
