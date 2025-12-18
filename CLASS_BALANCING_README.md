# Class Balancing in Training Dataset - December 2, 2025

## Changes Made

Modified `train_yolov11x_unified.py` to ensure **equal class distribution** in the training dataset by downsampling larger classes to match the smallest class.

## Problem Solved

Previously, if one class (e.g., fluorescent tube) had significantly more images than another class (e.g., outlet), the model would be trained on an imbalanced dataset. This can lead to:
- Bias toward the majority class
- Poor performance on minority classes
- Unequal learning across classes

## Solution Implemented

### Step 1: Collect Images by Class
Instead of collecting all images into one list, images are now organized by class ID:
```python
class_image_data = {
    0: [],  # fluorescent tube
    1: [],  # fire extinguisher
    2: []   # outlet
}
```

### Step 2: Find Minimum Class Size
Determine which class has the fewest images:
```python
min_class_size = min(len(class_image_data[0]), 
                     len(class_image_data[1]), 
                     len(class_image_data[2]))
```

### Step 3: Downsample to Match Minimum
Randomly sample from larger classes to match the smallest class:
```python
for class_id in [0, 1, 2]:
    class_images = class_image_data[class_id]
    if len(class_images) > min_class_size:
        # Randomly sample to match min_class_size
        sampled_images = random.sample(class_images, min_class_size)
        all_image_data.extend(sampled_images)
    else:
        # Use all images if already at or below min
        all_image_data.extend(class_images)
```

### Step 4: Shuffle and Split
After balancing, shuffle all images and split 60/20/20:
- This ensures each split has roughly equal representation of all classes
- Random shuffling prevents any ordering bias

## Example Output

```
📊 CLASS DISTRIBUTION (Before Balancing):
============================================================
   fluorescent tube: 450 images
   fire extinguisher: 280 images
   outlet: 195 images

⚖️ BALANCING CLASSES:
============================================================
   Smallest class has 195 images
   Randomly sampling all classes to match this size...

   fluorescent tube: Sampled 195 from 450 images
   fire extinguisher: Sampled 195 from 280 images
   outlet: Using all 195 images

✅ Balanced dataset created: 585 images total
   195 images per class × 3 classes = 585 images
============================================================

📊 CLASS DISTRIBUTION PER SPLIT:
============================================================
   Train:
      fluorescent tube: 117 images
      fire extinguisher: 117 images
      outlet: 117 images
   Valid:
      fluorescent tube: 39 images
      fire extinguisher: 39 images
      outlet: 39 images
   Test:
      fluorescent tube: 39 images
      fire extinguisher: 39 images
      outlet: 39 images
============================================================
```

## Benefits

1. **Equal Class Representation**: Each class has exactly the same number of images
2. **Unbiased Training**: Model learns all classes equally
3. **Better Generalization**: Prevents overfitting to majority class
4. **Consistent Performance**: All classes get equal attention during training
5. **Reproducible**: Uses `random.seed(42)` for consistent sampling

## Trade-offs

- **Reduced Dataset Size**: If one class has significantly fewer images, the total dataset will be smaller
- **Data Utilization**: Some images from larger classes won't be used
- **Benefit**: However, balanced training often produces better overall model performance than using all data with imbalance

## Alternative Approaches Considered

1. **Oversampling minority classes**: Duplicate images from smaller classes
   - Downside: Can lead to overfitting on minority classes
   
2. **Class weights**: Use weighted loss to give more importance to minority classes
   - Downside: More complex, requires hyperparameter tuning
   
3. **Data augmentation**: Generate synthetic images for minority classes
   - Downside: Requires additional preprocessing, may not be representative

**Chosen approach (downsampling)** is simple, effective, and prevents overfitting while ensuring balanced learning.

## Usage

Run dataset creation as normal:
```bash
python train_yolov11x_unified.py
```

Or skip training to just create the balanced dataset:
```bash
python train_yolov11x_unified.py --skip-training
```

The script will automatically:
1. Scan all datasets
2. Show class distribution before balancing
3. Downsample to create balanced dataset
4. Show class distribution after balancing and in each split
5. Proceed with training

## Verification

The output now includes detailed class distribution reports:
- Before balancing: See total images per class
- After balancing: Confirm equal distribution
- Per split: Verify balanced distribution in train/valid/test sets

This ensures you can verify the balancing is working correctly!
