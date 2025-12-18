from ollama import chat
import base64
import os

content = open("prompt.txt").read()
file_number = 0


for file in os.listdir("inspection images"):
    file_number += 1
    print("image:", file_number)
    file_path = os.path.join("inspection images", file)
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    message = {
    "role": "user",
    "content": content,
    #"content": "is the lightbulb broken or not",
    "images": [b64]
    }

    resp = chat(model="llava:7b", messages=[message])
    response = resp["message"]["content"]

    print(response)

