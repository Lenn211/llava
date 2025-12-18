from ollama import chat
import base64

with open("lightbulb1.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

message = {
  "role": "user",
  "content": "if the light appears broken, print 'broken'."
  "If the light appears in good condition, print 'okay'. Do not add any more words. "
  "Make sure your output consists of one single word.",
  #"content": "is the lightbulb broken or not",
  "images": [b64]
}

resp = chat(model="llava:7b", messages=[message])
response = resp["message"]["content"].strip().lower() 

print(response)

if response == 'broken':
    print('Robot turns to the right')
elif response == 'okay':
    print('robot turns to the left')
else:
    print(response)
