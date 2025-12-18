import cv2
from ultralytics import YOLO
import os
import glob
from PIL import Image
from ollama import chat
import base64
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define inspection prompts with synonyms for better detection
INSPECTION_ELEMENTS = {
    "sockets": ["wall socket", "power outlet", "electrical socket", "power point", "wall outlet"],
    "fire_safety": ["fire extinguisher", "fire safety equipment", "fire suppression device"],
    "lighting": ["light fixture", "ceiling light", "wall light", "lamp", "light fitting"]
}

def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def crop_object(image_path, box, padding=10):
    """Crop the detected object from the image with optional padding"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Extract coordinates and add padding
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    
    # Crop the image
    cropped = img[y1:y2, x1:x2]
    return cropped

def get_element_category(detected_label):
    """Map a detected label back to its category"""
    detected_label = detected_label.lower()
    for category, prompts in INSPECTION_ELEMENTS.items():
        if any(prompt.lower() in detected_label for prompt in prompts):
            return category
    return None

def inspect_element_llava(element_name, image_path, crop_box=None):
    """Use LLaVA to inspect the condition of a detected element"""
    print(f"🤖 Inspecting {element_name} up close...")
    
    crop_path = None
    if crop_box is not None:
        # Crop the image to the detected object
        cropped_img = crop_object(image_path, crop_box)
        if cropped_img is not None:
            # Save the cropped image permanently for inspection
            crop_path = os.path.join('zero_shot_results', f'cropped_{element_name}_{os.path.basename(image_path)}')
            cv2.imwrite(crop_path, cropped_img)
            print(f"  📸 Cropped image saved to: {crop_path}")
            image_to_inspect = crop_path
        else:
            image_to_inspect = image_path
    else:
        image_to_inspect = image_path

    b64 = encode_image(image_to_inspect)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an inspection robot. You must ONLY respond in one of these two formats:\n"
                "1. 'OK: [element] in good condition' if no issues are found\n"
                "2. 'FAULTY: [brief reason]' if you spot any defects\n"
                "No other response format is allowed. No explanations or commentary."
            )
        },
        {
            "role": "user",
            "content": f"Inspect this {element_name}. Is it in good condition or faulty? Remember to use ONLY the required response format.",
            "images": [b64]
        }
    ]
    
    try:
        resp = chat(model="llava:7b", messages=messages)
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"Error during LLaVA inspection: {e}")
        return "ERROR: Unable to complete inspection"

def detect_and_save(model, image_path, save_dir='zero_shot_results', device=0):
    """Run detection and save annotated image"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Run prediction with GPU device
    results = model.predict(source=image_path, conf=0.10, verbose=False, device=device)

    # Process results
    detections = []
    if results and len(results) > 0:
        result = results[0]  # Get first result
        img = cv2.imread(image_path)
        
        for box in result.boxes:
            # Get detection info
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
            
            # Store detection
            detections.append({
                "label": class_name,
                "confidence": float(confidence),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Save annotated image
        output_path = os.path.join(save_dir, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to: {output_path}")
    
    return detections

def main():
    # Source image to inspect
    image_path = "examples/example_1 copy.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        return

    print("🔍 Starting automated inspection...")
    
    # Check GPU availability and set device
    device = 0 if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    else:
        print("⚠️ GPU not available, using CPU")
    
    # Combine all prompts for a single detection run
    all_prompts = []
    for prompts in INSPECTION_ELEMENTS.values():
        all_prompts.extend(prompts)
    
    print(f"\n📋 Scanning for all elements: {', '.join(all_prompts)}...")
    
    # Initialize YOLO-World model with GPU
    model = YOLO("yolov8l-world.pt")
    model.set_classes(all_prompts)
    
    # Run detection with GPU
    detections = detect_and_save(model, image_path, device=device)
    
    # Clear memory after detection
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if not detections:
        print("No elements detected in the image.")
        return

    # Group detections by category
    categorized_detections = {}
    
    # Process each detection and categorize it
    print("\n🔎 Starting detailed inspections...")
    for detection in detections:
        element_name = detection["label"]
        category = get_element_category(element_name)
        
        if category and category not in categorized_detections:
            categorized_detections[category] = {
                "category": category,
                "image_path": image_path,  # Use original image path
                "detected_label": element_name,
                "confidence": detection["confidence"],
                "box": detection["box"]  # Store the bounding box for cropping
            }

    # Inspect each unique category found
    for info in categorized_detections.values():
        category = info["category"]
        
        print(f"\n🚗 Moving to inspect {category} (detected as: {info['detected_label']}, confidence: {info['confidence']:.2f})...")
        print("📍 Position reached. Starting detailed inspection...")
        
        # Use LLaVA to inspect the element's condition, passing the bounding box for cropping
        result = inspect_element_llava(category, info["image_path"], crop_box=info["box"])
        print(f"🔎 Inspection result: {result}")
        
        # Process the inspection result
        r = result.lower().strip()
        if r.startswith("ok") or "good condition" in r or r == "ok":
            print(f"✅ {category} marked as INSPECTED\n")
        elif r.startswith("faulty") or "defect" in r or "damage" in r or "fault" in r:
            # Extract the reason after "faulty:" if present
            reason = result[8:] if ":" in result else result
            print(f"⚠️ {category} marked as FAULTY – {reason}\n")
        else:
            print(f"❓ Unable to classify result: {result}\n")
        
        # Clear memory after each inspection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()