from ollama import chat
import base64
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection, AutoProcessor, AutoModelForZeroShotObjectDetection
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

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
# Load Grounding DINO & LLaVA
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

# Grounding DINO
processor_dino = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model_dino = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

# LLaVA processor (for inspection)
processor_llava = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model_llava = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

# -----------------------------
# Utility functions
# -----------------------------
def response_generator(messages):
    resp = chat(model="llava:7b", messages=messages)
    return resp["message"]["content"].strip()

def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def visualize_detections(image_path, detections, tag):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    for det in detections:
        box = det["box"]
        label = f"{det['label']} ({det['score']})"
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle((x0, y0), width, height, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, label, color='lime', fontsize=10, weight='bold')

    output_folder = "annotated_results"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"detections_{tag}.jpg")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Annotated image saved to: {output_path}")

# -----------------------------
# Detection with Grounding DINO
# -----------------------------
def detect_elements(image_path, batch_size=1):
    print(f"\n🔍 Detecting elements in {image_path}...")
    image = Image.open(image_path).convert("RGB")

    # Downsize image to max 800px on longer side to reduce VRAM usage
    max_dim = 800
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"🖼️ Image resized to {new_size} for efficient processing.")

    all_boxes = []
    all_scores = []
    all_labels = []

    # Prepare list of element names
    element_names = [v for k, v in inspection_elements.items()]

    # Helper to chunk the element list
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for batch in chunks(element_names, batch_size):
        images_batch = [image] * len(batch)
        try:
            # Tokenize / preprocess for the whole batch (padding=True to handle variable length)
            inputs = processor_dino(images=images_batch, text=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # move tensors to device

            # Target size repeated for each item in batch
            target_sizes = torch.tensor([image.size[::-1]] * len(batch)).to(device)

            # Inference with autocast to reduce memory (if CUDA available)
            with torch.no_grad():
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model_dino(**inputs)
                else:
                    outputs = model_dino(**inputs)

            # Post-process returns a list of results, one per batch item
            results_list = processor_dino.post_process_grounded_object_detection(
                outputs,
                target_sizes=target_sizes,
                input_ids=inputs.get("input_ids")
            )

            # Collect detections from each result in the batch
            for element_name, results in zip(batch, results_list):
                if results is None:
                    continue
                labels_key = "text_labels" if "text_labels" in results else "labels"
                for box, label, score in zip(results["boxes"], results[labels_key], results["scores"]):
                    if score.item() >= 0.35:
                        all_boxes.append(box.cpu())  # Move to CPU
                        all_scores.append(score.cpu())
                        all_labels.append({
                            "label": label if isinstance(label, str) else element_name,
                            "element_name": element_name,
                            "score": score.item(),
                            "box": box.cpu()
                        })
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("\n❌ CUDA out of memory! Try reducing image size or ensure no other GPU processes are running.")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                continue
            else:
                raise
        finally:
            try:
                del inputs
                del outputs
                del results_list
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        torch.cuda.empty_cache()

    if not all_boxes:
        print("❌ No elements detected.")
        return []

    # Move stacked tensors to the same device before NMS
    boxes_tensor = torch.stack(all_boxes).to(device)
    scores_tensor = torch.tensor([s.item() if isinstance(s, torch.Tensor) else float(s) for s in all_scores], device=device)

    # Non-Maximum Suppression to reduce overlaps
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
    detections = []
    for idx in keep_indices:
        info = all_labels[idx]
        detections.append({
            "element_name": info["element_name"],
            "label": info["label"],
            "score": round(info["score"], 3),
            "box": [round(v.item(), 2) for v in info["box"]]
        })

    print(f"✅ Detected {len(detections)} unique element(s).")
    return detections

# -----------------------------
# Step 2 – LLaVA inspection up close
# -----------------------------
def inspect_element_llava(element_name, example_2_path):
    print(f"🤖 Inspecting {element_name} up close...")
    b64 = encode_image(example_2_path)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an inspection robot analyzing a close-up image of a {element_name}. "
                "Respond with one of the following formats only:\n"
                "- 'OK: [Element] in good condition'\n"
                "- 'FAULTY: [brief reason]'\n"
                "Do not add any extra commentary."
            )
        },
        {
            "role": "user",
            "content": f"Inspect the {element_name} closely for defects or abnormalities.",
            "images": [b64]
        }
    ]
    return response_generator(messages)

# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    example_1 = "example_2.jpg"  # initial overview
    example_2 = "example_2.jpg"  # close-up after driving

    # STEP 1: Detect all elements
    detections = detect_elements(example_1)
    visualize_detections(example_1, detections, "overview")
    
    torch.cuda.empty_cache()
    if not detections:
        exit()

    # STEP 2: Sort elements alphabetically
    detections_sorted = sorted(detections, key=lambda d: d["element_name"].lower())

    # STEP 3: Inspect each element one by one
    for det in detections_sorted:
        elem = det["element_name"]
        print(f"\n🚗 Driving towards {elem}...")
        print(f"📍 Position reached. Capturing close-up image...")

        result = inspect_element_llava(elem, example_2)
        print(f"🔎 LLaVA Output: {result}")

        torch.cuda.empty_cache()  # Clear after each inspection

        r = result.lower()
        if r.startswith("ok:"):
            print(f"✅ {elem} marked as INSPECTED\n")
        elif r.startswith("faulty:"):
            print(f"⚠️ {elem} marked as FAULTY – {result[8:]}\n")
        else:
            print(f"❓ Unable to classify result: {result}\n")
