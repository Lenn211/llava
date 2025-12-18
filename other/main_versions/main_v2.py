from ollama import chat
import base64
import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# -----------------------------
# Define inspection elements
# -----------------------------
inspection_elements = {
    "1": "Facade",
    "2": "Windows",
    "3": "Pathways",
    "4": "Lights",
    "5": "Doors",
    "6": "Common Areas",
    "7": "Fire Extinguishers",
    "8": "Socket"
}

# -----------------------------
# Load Grounding DINO model once
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

# -----------------------------
# Utility functions
# -----------------------------
def response_generator(messages):
    resp = chat(model="llava:13b", messages=messages)
    return resp["message"]["content"].strip()

def load_image(number=5):
    folder = "inspection images"
    file_list = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not file_list:
        raise FileNotFoundError("No images found in inspection folder.")
    file_path = os.path.join(folder, file_list[number])
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, file_path

def pretraining_examples_1():
    return [
        {"file": "IMG-20251001-WA0002.jpg", "expected_response": "4"},
        {"file": "IMG-20251001-WA0003.jpg", "expected_response": "7"},
        {"file": "IMG-20251001-WA0001.jpg", "expected_response": "7"},
    ]

# -----------------------------
# Step 1 – Inspection with LLaVA
# -----------------------------
def step_1_message(b64, use_pretraining=True):
    messages = []
    if use_pretraining:
        for ex in pretraining_examples_1():
            example_path = os.path.join("Examples", ex["file"])
            if os.path.exists(example_path):
                with open(example_path, "rb") as f:
                    b64_ex = base64.b64encode(f.read()).decode("utf-8")
                messages.append({
                    "role": "user",
                    "content": f"Example inspection image: {ex['file']}",
                    "images": [b64_ex]
                })
                messages.append({
                    "role": "assistant",
                    "content": ex["expected_response"]
                })

    messages.extend([
        {
            "role": "system",
            "content": (
                "You are an autonomous inspection robot. "
                "You must visually analyze the image to make decisions. "
                "Do not describe or explain what you see — only output the inspection outcome. "
                "You have a numbered list of inspection elements to reference. "
                "Your only allowed outputs are: '0' or '[element_number]'. "
                "Never output anything else.\n\n"
                "Inspection element reference list:\n"
                + "\n".join([f'{k}: {v}' for k, v in inspection_elements.items()])
            )
        },
        {
            "role": "user",
            "content": (
                "Step 1: INSPECT.\n\n"
                "Inspect all visible elements in the image.\n"
                "If all visible elements are in good condition, output exactly: 0.\n"
                "If any element appears unclear, damaged, or less than good, output exactly: [element_number].\n\n"
                "The element_number must correspond to the element that determined your decision.\n"
                "Only output the number(s) — for example: 1 or 2-3."
            ),
            "images": [b64]
        }
    ])
    return messages

# -----------------------------
# Draw bounding boxes
# -----------------------------
def visualize_detections(image_path, detections, element_number):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    for det in detections:
        box = det["box"]
        label = f"{det['label']} ({det['score']})"
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, label, color='lime', fontsize=10, weight='bold')

    output_folder = "annotated_results"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"element_{element_number}_detections.jpg")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Annotated image saved to: {output_path}")

# -----------------------------
# Step 2 – Locate element with Grounding DINO
# -----------------------------
import torch
from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

# Load once globally (faster)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

import torch
from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

# Load once globally (for performance)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

def locate_element_grounding_dino(image_path, element_number):
    detection_queries = {
        "1": "building facade, wall, exterior wall",
        "2": "window, glass window, door, doorway",
        "3": "pathway, sidewalk, floor, ground path",
        "4": "light, lamp, ceiling light, street light",
        "5": "door, wooden door, metal door",
        "6": "common area, hallway, lobby, corridor",
        "7": "fire extinguisher, safety extinguisher, red cylinder",
        "8": "electrical outlet, wall socket, power plug, socket, plug socket"
    }

    image = Image.open(image_path).convert("RGB")
    text_query = detection_queries.get(str(element_number), "object")

    # Prepare inputs for GroundingDINO
    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)

    # ✅ Correct post-processing — no thresholds here
    results = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes)[0]

    # ✅ Apply thresholds manually
    score_threshold = 0.3
    detections = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if score.item() >= score_threshold:
            detections.append({
                "label": label,
                "score": round(score.item(), 3),
                "box": [round(v.item(), 2) for v in box]
            })

    if not detections:
        print("No confident detections found for:", text_query)
    return detections



# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    b64, image_path = load_image(5)

    response = response_generator(step_1_message(b64))
    print("Step 1 Response:", response)

    if response != "0" or response =="0":
        element_number = "8"    #response.strip()
        print(f"Step 2: Locate element {element_number} — {inspection_elements[element_number]}")
        detections = locate_element_grounding_dino(image_path, element_number)

        if detections:
            print("Detected locations:")
            for det in detections:
                print(f" - {det['label']} at {det['box']} (confidence: {det['score']})")
            visualize_detections(image_path, detections, element_number)
        else:
            print("No matching object found.")
    else:
        print("All elements in good condition. Continue driving.")
