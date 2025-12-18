# Retrain Custom Model with ALL Synonym Classes

## Overview
This will retrain your YOLOv8-World model to recognize ALL synonym variations as trained classes.

## Classes to Train (16 total)

### Sockets (6 classes)
1. outlet
2. power outlet
3. wall socket
4. electrical socket
5. power point
6. wall outlet

### Fire Safety (3 classes)
7. fire extinguisher
8. fire safety equipment
9. fire suppression device

### Lighting (7 classes)
10. fluorescent tube
11. light fixture
12. ceiling light
13. wall light
14. lamp
15. light fitting
16. fluorescent light

## Step 1: Update data.yaml

Edit `unified_dataset/data.yaml`:

```yaml
names:
  - outlet
  - power outlet
  - wall socket
  - electrical socket
  - power point
  - wall outlet
  - fire extinguisher
  - fire safety equipment
  - fire suppression device
  - fluorescent tube
  - light fixture
  - ceiling light
  - wall light
  - lamp
  - light fitting
  - fluorescent light
nc: 16
train: /path/to/unified_dataset/train/images
val: /path/to/unified_dataset/valid/images
```

## Step 2: Relabel Training Data

You need to relabel ALL your training images to use ALL 16 classes.

**Current labels:**
- Class 0: outlet (maps to outlet)
- Class 1: fire extinguisher (maps to fire extinguisher)
- Class 2: fluorescent tube (maps to fluorescent tube)

**New labels needed:**
Each outlet can be labeled as:
- Class 0: outlet
- OR Class 1: power outlet
- OR Class 2: wall socket
- etc.

**Problem**: This requires manually deciding which synonym to use for each instance.

## Step 3: Train the Model

```bash
python train_yolov8x_world_unified.py --epochs 100 --batch 4
```

## MAJOR PROBLEMS with This Approach

### 1. **Duplicate Detections**
The model will detect the same object multiple times:
- Detection 1: "outlet" (conf: 0.85)
- Detection 2: "power outlet" (conf: 0.78)
- Detection 3: "wall socket" (conf: 0.65)

All pointing to the SAME physical outlet!

### 2. **Training Complexity**
- You need to decide which label to use for each image
- Inconsistent labeling will confuse the model
- Much longer training time (16 classes vs 3)

### 3. **Lower Accuracy**
- Model has to learn 16 classes instead of 3
- More confusion between similar classes
- Needs MORE training data

## RECOMMENDED SOLUTION

**Keep the current approach:**

✅ **Use Custom Model for Accuracy**
- Detects: `outlet`, `fire extinguisher`, `fluorescent tube`
- Fast, accurate, works well

✅ **Use Base YOLOv8-World for Flexibility**
- Detects: All synonyms
- Slower, less accurate, but very flexible

✅ **Let Users Choose**
- Your GUI already has radio buttons!
- "Custom YOLOv8x" = fast & accurate for known objects
- "YOLOv8x-World" = flexible for any synonym

## Alternative: Hybrid Approach

Use BOTH models together:
1. Run custom model first (fast, accurate)
2. If no detections, run base model with synonyms (flexible)

Would you like me to implement this hybrid approach?
