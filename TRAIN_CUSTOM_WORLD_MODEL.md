# Training a Custom YOLOv8x-World Model with Synonym Support

## Current Situation
- Your custom model `custom_yolov8x.pt` is trained with 3 classes: `outlet`, `fire extinguisher`, `fluorescent tube`
- You want it to also recognize synonyms like "power outlet", "wall socket", "fire safety equipment", etc.

## The Solution: YOLOv8-World

**Standard YOLOv8** models cannot recognize synonyms - they only detect the exact class names they were trained on.

**YOLOv8-World** models have open-vocabulary detection, meaning:
1. Train on base classes: `outlet`, `fire extinguisher`, `fluorescent tube`
2. At inference time, use ANY synonyms via `model.set_classes()`

## How to Train YOLOv8x-World

### Step 1: Verify Your Dataset
Your unified dataset should be at:
```
unified_dataset/
  ├── data.yaml
  ├── train/
  │   ├── images/
  │   └── labels/
  └── valid/
      ├── images/
      └── labels/
```

### Step 2: Train the Model
Run your existing training script:
```bash
python train_yolov8x_world_unified.py --epochs 100 --batch 4 --imgsz 640
```

This will create a YOLOv8x-World model that:
- Is trained on your 3 base classes
- Can detect using ANY synonyms at inference time

### Step 3: Rename the Output
After training completes, the best weights will be at:
```
runs/detect/yolov8x_world_unified/weights/best.pt
```

Copy it to your main directory:
```bash
cp runs/detect/yolov8x_world_unified/weights/best.pt custom_yolov8x_world.pt
```

### Step 4: Update the GUI
Change the model filename in `main_gemini_gui.py` line 189:
```python
value="custom_yolov8x_world.pt",  # Add "_world" to the filename
```

## Why This Works

**YOLOv8-World Architecture**:
- Uses CLIP-style vision-language embeddings
- Learns to match visual features with text descriptions
- Can generalize to synonyms without retraining

**Example**:
- Trained on: `outlet`
- Can detect: "power outlet", "wall socket", "electrical socket" (semantically similar)
- Trained on: `fire extinguisher`
- Can detect: "fire safety equipment", "fire suppression device" (semantically similar)

## Verify Your Current Model

To check if your current `custom_yolov8x.pt` is a World model, run:
```python
from ultralytics import YOLO
model = YOLO("custom_yolov8x.pt")
print(model.model)  # Should show "WorldModel" in the architecture
```

If it shows standard YOLOv8, you need to retrain with the `-world` variant.

## Expected Results

After training YOLOv8x-World:
- ✅ Detects `outlet` and all synonyms (power outlet, wall socket, etc.)
- ✅ Detects `fire extinguisher` and synonyms (fire safety equipment, etc.)
- ✅ Detects `fluorescent tube` and synonyms (light fixture, ceiling light, etc.)
- ✅ Higher accuracy than base YOLOv8-World (trained on your specific data)
- ✅ Faster inference than YOLOv8-World (smaller vocabulary)

## Training Tips

1. **Batch Size**: Start with 4-8 depending on your GPU
2. **Epochs**: 100-200 for good convergence
3. **Image Size**: 640 is standard, increase to 1280 for better accuracy (slower)
4. **Patience**: Training takes several hours on a good GPU

## Troubleshooting

**Q: My custom model still doesn't detect synonyms**
A: Verify it's a YOLOv8-World model, not standard YOLOv8

**Q: Can I train standard YOLOv8 with all synonyms?**
A: No - it would treat each synonym as a separate class, causing duplicate detections

**Q: Should I use YOLOv8-World or custom YOLOv8-World?**
A: Custom is better - it's trained on your data for higher accuracy
