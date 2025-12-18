import cv2
from ultralytics import YOLO
import os
import glob
from PIL import Image
import base64
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import requests
import json
import io
import time

# ============================================================
# CONFIGURATION: Set your Gemini API key here
# ============================================================
# Option 1: Set your API key directly here (recommended for simplicity)
GEMINI_API_KEY = "AIzaSyCdqwK47KRGvnPDpxNzH6EkicBFGaAeFKE"  # Replace with your actual API key

# Option 2: Leave as None to use environment variable GEMINI_API_KEY
# GEMINI_API_KEY = None
# ============================================================

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

def inspect_element_gemini(element_name, image_path, crop_box=None, api_key=None):
    """Use Gemini 2.5 Flash to inspect the condition of a detected element via REST API"""
    print(f"🤖 Inspecting {element_name} up close with Gemini 2.5 Flash...")
    
    # Get API key: use passed parameter, then global config, then environment variable
    if not api_key:
        api_key = GEMINI_API_KEY if GEMINI_API_KEY and GEMINI_API_KEY != "your-api-key-here" else None
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("⚠️ Warning: GEMINI_API_KEY not found. Please set it in the script or as an environment variable.")
        return "ERROR: API key not configured"
    
    crop_path = None
    if crop_box is not None:
        # Crop the image to the detected object
        cropped_img = crop_object(image_path, crop_box)
        if cropped_img is not None:
            # Save the cropped image permanently for inspection 
            crop_path = os.path.join('gemini_results', f'cropped_{element_name}_{os.path.basename(image_path)}')
            os.makedirs('gemini_results', exist_ok=True)
            cv2.imwrite(crop_path, cropped_img)
            print(f"  📸 Cropped image saved to: {crop_path}")
            image_to_inspect = crop_path
        else:
            image_to_inspect = image_path
    else:
        image_to_inspect = image_path

    # Load and encode image for Gemini REST API
    with open(image_to_inspect, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Create the prompt
    prompt = f"""You are an inspection robot. You must ONLY respond in one of these two formats:
1. 'OK: [element] in good condition' if no issues are found
2. 'FAULTY: [brief reason]' if you spot any defects

No other response format is allowed. No explanations or commentary.

Inspect this {element_name}. Is it in good condition or faulty? Remember to use ONLY the required response format."""

    try:
        # Gemini REST API endpoint (using gemini-1.5-flash model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        
        # Prepare request payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }]
        }
        
        headers = {'Content-Type': 'application/json'}
        
        # Make the API request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text.strip()
        else:
            return "ERROR: No response from Gemini API"
            
    except requests.exceptions.RequestException as e:
        print(f"Error during Gemini inspection: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return "ERROR: Unable to complete inspection"
    except Exception as e:
        print(f"Error during Gemini inspection: {e}")
        return "ERROR: Unable to complete inspection"

def detect_and_save(model, image_path, save_dir='gemini_results', device=0):
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
    # Get all images from the gemini_test_folder
    test_images_dir = "gemini_test_folder"
    if not os.path.exists(test_images_dir):
        print(f"Error: '{test_images_dir}' folder not found!")
        return
    
    # Find all image files in the test images folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    if not image_paths:
        print(f"Error: No images found in '{test_images_dir}' folder!")
        return
    
    print(f"📁 Found {len(image_paths)} images in '{test_images_dir}' folder")
    print("🔍 Starting automated inspection with Gemini 2.5 Flash...\n")
    
    # Check for Gemini API key
    api_key = GEMINI_API_KEY if GEMINI_API_KEY and GEMINI_API_KEY != "your-api-key-here" else None
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("⚠️ GEMINI_API_KEY not found in script or environment variable!")
        print("Please set it in the script or using: export GEMINI_API_KEY='your-api-key-here'")
        return
    
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
    
    print(f"📋 Scanning for all elements: {', '.join(all_prompts)}...\n")
    
    # Process each image in the folder
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*80}")
        print(f"📷 Processing image {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
        print(f"{'='*80}\n")
        
        # Initialize YOLO-World model with GPU and set classes
        model = YOLO("yolov8x-world.pt")
        model.set_classes(all_prompts)  # Set the classes for YOLO-World to detect
        
        # Run detection with GPU
        detections = detect_and_save(model, image_path, device=device)
        
        # Clear memory after detection
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if not detections:
            print("⚠️ No elements detected in this image.")
            continue

        # Group detections by category
        categorized_detections = {}
        
        # Process each detection and categorize it
        print(f"\n🔎 Found {len(detections)} detection(s). Starting detailed inspections with Gemini 2.5 Flash...")
        for detection in detections:
            element_name = detection["label"]
            category = get_element_category(element_name)
            
            print(f"   Detected: '{element_name}' → Category: '{category}'")
            
            # If we can't categorize it, use the label itself as the category
            if not category:
                category = element_name
            
            if category not in categorized_detections:
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
            
            # Use Gemini to inspect the element's condition, passing the bounding box for cropping
            result = inspect_element_gemini(category, info["image_path"], crop_box=info["box"], api_key=api_key)
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
        
        # Wait 5 seconds before processing next image to avoid rate limiting
        if idx < len(image_paths):
            print(f"\n⏳ Waiting 5 seconds before processing next image...")
            time.sleep(10)
    
    print(f"\n{'='*80}")
    print(f"✅ Completed inspection of all {len(image_paths)} images!")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
