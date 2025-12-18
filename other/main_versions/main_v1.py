from ollama import chat
import base64
import os
import json


inspection_elements = {
    "1": "Walls and Facade",
    "2": "Windows and Doors",
    "3": "Pathways",
    "4": "Lighting Maintenance",
    "5": "Leak Detection",
    "6": "Common Areas",
    "7": "Fire Extinguishers",
    "8": "Socket Inspection"
}

def load_prompt():
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    content = prompt_data.get("content", {})
    purpose = content.get("purpose", "")
    step_1 = content.get("step 1", {}).get("rules", [])
    return purpose, step_1, content


def response_generator(messages):
    resp = chat(model="llava:13b", messages=messages)
    return resp["message"]["content"].strip()


def pretraining_examples_1():
    examples = [
        {
            "file": "IMG-20251001-WA0002.jpg",
            "expected_response": "4"
        },
        {
            "file": "IMG-20251001-WA0003.jpg",
            "expected_response": "7"
        },
        {
            "file": "IMG-20251001-WA0001.jpg",
            "expected_response":  "0"
        }
    ]
    return examples


def step_1_message(b64, use_pretraining=True):
    """Generate the Step 1 (Inspect) message sequence, optionally including pretraining examples."""
    
    messages = []

    #  1. Include pretraining examples (few-shot)
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


    #  2. Add main system + user messages for the current inspection
    messages.extend([
        {
            "role": "system",
            "content": (
                "You are an autonomous inspection robot. "
                "You must visually analyze the image to make decisions. "
                "Do not describe or explain what you see — only use your analysis to determine the correct inspection outcome. "
                "You have a numbered list of inspection elements you may reference. "
                "Your only allowed outputs are: '0' or '[element_number]'. "
                "Never output anything else.\n\n"
                "Inspection element reference list:\n"
                + "\n".join([f"{k}: {v}" for k, v in inspection_elements.items()])
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
                "Do not explain your reasoning. Only output the exact format — for example:\n"
                "1 or 2-3."
            ),
            "images": [b64]
        }
    ])

    return messages



def step_2_message(element_number, b64):
    """Generate the Step 2 (Locate) message sequence for a specific element."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous inspection robot. "
                "You must identify the coordinates of a particular object in the image. "
                "Do not describe or explain what you see — only output the coordinates of the object. "
                "Your only allowed outputs are: 'x1,y1,x2,y2' (the bounding box coordinates of the object) "
                "or 'N/A' if the object is not found.\n\n"
                "Inspection elements (reference by number):\n"
                + "\n".join([f"{k}: {v}" for k, v in inspection_elements.items()])
            )
        },
        {
            "role": "user",
            "content": (
                f"Step 2: LOCATE.\n\n"
                f"Locate the element number {element_number} ({inspection_elements[str(element_number)]}).\n"
                "Output only the bounding box coordinates (x1,y1,x2,y2) if found, or 'N/A' if not found.\n"
                "No explanations, no descriptions — only output the coordinates or N/A."
            ),
            "images": [b64]
        }
    ]
    return messages


# Inspect one image only (first image in folder)
def load_image(number=4):
    folder = "inspection images"
    file_list = os.listdir(folder)
    if not file_list:
        raise FileNotFoundError("No images found in inspection folder.")

    file_path = os.path.join(folder, file_list[number])
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64


response = response_generator(step_1_message(load_image(5), use_pretraining=True))
print(response)

if response != "0":
    element_number = response
    print(f"Element {element_number} requires further inspection.")
    messages = step_2_message(element_number, load_image())
    response = response_generator(messages)
    print(response)
elif response == "0":
    print("All elements in good condition. Continue driving.")
else:
    print("Unexpected response:", response)