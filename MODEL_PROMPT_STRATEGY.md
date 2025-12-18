# Model Prompt Strategy

## Overview
The GUI now uses different detection prompts for each model type to maximize detection accuracy while respecting each model's capabilities.

## Prompt Sets

### Base YOLOv8x-World Model (`yolov8x-world.pt`)
**Strategy:** Open-vocabulary detection with synonyms

```python
INSPECTION_ELEMENTS_BASE = {
    "sockets": ["wall socket", "power outlet", "electrical socket", "power point", "wall outlet"],
    "fire_safety": ["fire extinguisher", "fire safety equipment", "fire suppression device"],
    "lighting": ["light fixture", "ceiling light", "wall light", "lamp", "light fitting", "fluorescent tube"]
}
```

**Why these prompts?**
- The base model has open-vocabulary capabilities
- Can detect objects using natural language descriptions
- Benefits from synonym variations
- Uses `model.set_classes()` to set detection prompts dynamically

**Total prompts:** 14 different terms across 3 categories

---

### Custom Fine-Tuned Model (`custom_yolov8x.pt`)
**Strategy:** Use trained class names only

```python
INSPECTION_ELEMENTS_CUSTOM = {
    "sockets": ["outlet"],
    "fire_safety": ["fire extinguisher"],
    "lighting": ["fluorescent tube"]
}
```

**Why these prompts?**
- The custom model was fine-tuned on specific class names
- Fine-tuning breaks open-vocabulary synonym matching
- Best performance with exact trained class names
- Does NOT use `model.set_classes()` - relies on native trained classes

**Total prompts:** 3 exact class names

---

## Implementation Details

### Automatic Prompt Selection
The GUI automatically selects the correct prompt set based on the chosen model:

```python
if "custom" in model_path.lower():
    INSPECTION_ELEMENTS = INSPECTION_ELEMENTS_CUSTOM
else:
    INSPECTION_ELEMENTS = INSPECTION_ELEMENTS_BASE
```

### Detection Configuration

**For Base Model:**
```python
model = YOLO("yolov8x-world.pt")
model.set_classes(all_prompts)  # Set open-vocabulary prompts
```

**For Custom Model:**
```python
model = YOLO("custom_yolov8x.pt")
# No set_classes() - uses native trained classes
```

### Category Mapping
The `get_element_category()` method checks both prompt sets to map detected labels to categories, ensuring it works regardless of which model was used.

---

## Expected Behavior

### When Using Base Model:
- ✅ Detects "wall socket", "power outlet", "electrical socket", etc.
- ✅ Detects "light fixture", "ceiling light", "fluorescent tube", etc.
- ✅ Open-vocabulary flexibility
- ⚠️ May have lower precision on specific object types

### When Using Custom Model:
- ✅ Detects "outlet" (trained class)
- ✅ Detects "fire extinguisher" (trained class)
- ✅ Detects "fluorescent tube" (trained class)
- ✅ Higher precision on these specific classes
- ❌ Does NOT detect synonyms like "wall socket" or "power outlet"
- ❌ Open-vocabulary capability lost after fine-tuning

---

## Key Advantages

1. **Base Model**: Maximum flexibility, general-purpose detection
2. **Custom Model**: Maximum accuracy on specific trained objects
3. **Automatic Selection**: No manual configuration needed
4. **Both Detect Same Objects**: Just using different vocabulary

---

## Testing Recommendations

### Test with Base Model:
```bash
# Should detect using various synonyms
python main_gemini_gui.py
# Select: yolov8x-world.pt
# Load test images with outlets, fire extinguishers, lights
```

### Test with Custom Model:
```bash
# Should detect using exact class names
python main_gemini_gui.py
# Select: custom_yolov8x.pt
# Load same test images
```

### Compare Results:
- Both models should detect the same physical objects
- Detection labels will differ (synonyms vs. class names)
- Custom model may show higher confidence/precision
- Base model may detect additional object types

---

## Future Enhancements

### Hybrid Detection (Optional)
You could implement a fallback strategy:
1. Try custom model first (high precision)
2. If no detections, try base model with synonyms (high recall)
3. Combine results

### Example:
```python
custom_detections = detect_with_custom_model()
if len(custom_detections) == 0:
    base_detections = detect_with_base_model()
    return base_detections
return custom_detections
```

This would give you "best of both worlds" - precision when available, recall as fallback.
