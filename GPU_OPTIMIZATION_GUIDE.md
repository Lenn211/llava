# GPU Optimization Guide for YOLOv11 Training

## 🎯 Quick Solutions to Lower GPU Load

### 1. **Model Size** (MOST IMPACTFUL)
Choose a smaller model architecture:

```bash
# Lowest GPU usage (~2-4 GB VRAM) ✅ RECOMMENDED FOR 4GB GPUS
python train_from_scratch_grayscale.py --model-size n --batch 2 --imgsz 416

# Low GPU usage (~4-6 GB VRAM) ✅ GOOD BALANCE
python train_from_scratch_grayscale.py --model-size s --batch 4 --imgsz 512

# Medium GPU usage (~6-8 GB VRAM)
python train_from_scratch_grayscale.py --model-size m --batch 4 --imgsz 640

# High GPU usage (~8-12 GB VRAM)
python train_from_scratch_grayscale.py --model-size l --batch 8 --imgsz 640

# Highest GPU usage (~12-16 GB VRAM) ⚠️ HIGH-END GPUs ONLY
python train_from_scratch_grayscale.py --model-size x --batch 8 --imgsz 640
```

**Model Size Comparison:**
- `n` (nano): 2.7M parameters - Fastest, lowest VRAM
- `s` (small): 11.1M parameters - Good balance
- `m` (medium): 25.9M parameters - Better accuracy
- `l` (large): 43.7M parameters - High accuracy
- `x` (xlarge): 68.2M parameters - Best accuracy, highest VRAM

### 2. **Reduce Batch Size**
Lower batch size = less GPU memory:

```bash
# For 4GB GPU
--batch 1

# For 6GB GPU
--batch 2

# For 8GB GPU
--batch 4

# For 12GB+ GPU
--batch 8
```

### 3. **Reduce Image Size**
Smaller images = less memory:

```bash
# Minimum (lowest GPU usage)
--imgsz 320

# Low
--imgsz 416

# Medium (default)
--imgsz 512

# High
--imgsz 640

# Very high (not recommended for training from scratch)
--imgsz 800
```

### 4. **Combined Low-GPU Configuration**

For **4GB GPU** (GTX 1650, RTX 3050 4GB, etc.):
```bash
python train_from_scratch_grayscale.py \
    --model-size n \
    --batch 1 \
    --imgsz 320 \
    --epochs 200
```

For **6GB GPU** (GTX 1060, RTX 2060, RTX 3050 6GB, etc.):
```bash
python train_from_scratch_grayscale.py \
    --model-size s \
    --batch 2 \
    --imgsz 416 \
    --epochs 200
```

For **8GB GPU** (RTX 3060 Ti, RTX 2070, etc.):
```bash
python train_from_scratch_grayscale.py \
    --model-size s \
    --batch 4 \
    --imgsz 512 \
    --epochs 200
```

For **12GB+ GPU** (RTX 3080, RTX 4070 Ti, etc.):
```bash
python train_from_scratch_grayscale.py \
    --model-size m \
    --batch 8 \
    --imgsz 640 \
    --epochs 200
```

---

## 🔧 Additional Optimizations

### 5. **Clear GPU Cache Before Training**
```bash
# Run this before training to free up GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

### 6. **Monitor GPU Usage During Training**
```bash
# In another terminal, run:
watch -n 1 nvidia-smi

# Or use:
nvidia-smi -l 1
```

### 7. **Close Other Programs**
- Close web browsers (especially Chrome with multiple tabs)
- Close Discord, Slack, or other communication apps
- Close video players
- Close any other Python processes

### 8. **Use Gradient Accumulation** (Advanced)
If you need larger effective batch size but have limited VRAM, modify the training code to use gradient accumulation (update weights every N batches instead of every batch).

---

## ⚡ Performance vs Accuracy Trade-offs

| Configuration | GPU RAM | Training Speed | Expected mAP | Use Case |
|--------------|---------|----------------|--------------|----------|
| nano + batch=1 + 320px | ~2GB | Fast | 0.6-0.7 | Quick testing, low-end GPU |
| nano + batch=2 + 416px | ~3GB | Fast | 0.65-0.75 | Low-end GPU, decent accuracy |
| small + batch=2 + 416px | ~4GB | Medium | 0.7-0.8 | Best balance for 6GB GPU |
| small + batch=4 + 512px | ~6GB | Medium | 0.75-0.85 | Good balance |
| medium + batch=4 + 640px | ~8GB | Slower | 0.8-0.9 | High accuracy |
| xlarge + batch=8 + 640px | ~14GB | Slowest | 0.85-0.95 | Maximum accuracy |

---

## 🚨 Troubleshooting Out of Memory (OOM) Errors

If you see `CUDA out of memory` errors:

1. **Reduce batch size by half**
   ```bash
   # If batch=4 fails, try batch=2
   # If batch=2 fails, try batch=1
   ```

2. **Reduce image size**
   ```bash
   # If 640 fails, try 512
   # If 512 fails, try 416
   # If 416 fails, try 320
   ```

3. **Use smaller model**
   ```bash
   # If 'm' fails, try 's'
   # If 's' fails, try 'n'
   ```

4. **Kill all other GPU processes**
   ```bash
   # Check what's using GPU
   nvidia-smi
   
   # Kill process by PID
   kill -9 <PID>
   ```

5. **Restart your computer** to fully clear GPU memory

---

## 📊 Recommended Configurations by GPU

### **2-4 GB VRAM** (GTX 1050 Ti, MX series)
```bash
python train_from_scratch_grayscale.py --model-size n --batch 1 --imgsz 320 --epochs 200
```
⚠️ Training from scratch on 2-4GB GPU is challenging. Consider using transfer learning instead.

### **4-6 GB VRAM** (GTX 1650, RTX 3050 4GB)
```bash
python train_from_scratch_grayscale.py --model-size n --batch 2 --imgsz 416 --epochs 200
```

### **6-8 GB VRAM** (GTX 1060, RTX 2060, RTX 3050 8GB)
```bash
python train_from_scratch_grayscale.py --model-size s --batch 4 --imgsz 512 --epochs 200
```

### **8-12 GB VRAM** (RTX 3060 Ti, RTX 2070 Super, RTX 4060)
```bash
python train_from_scratch_grayscale.py --model-size m --batch 4 --imgsz 640 --epochs 200
```

### **12-16 GB VRAM** (RTX 3080, RTX 4070 Ti, RTX 4080)
```bash
python train_from_scratch_grayscale.py --model-size l --batch 8 --imgsz 640 --epochs 200
```

### **16+ GB VRAM** (RTX 3090, RTX 4090, A100)
```bash
python train_from_scratch_grayscale.py --model-size x --batch 16 --imgsz 640 --epochs 200
```

---

## 💡 Pro Tips

1. **Start small, scale up**: Begin with nano model to verify everything works, then upgrade if needed

2. **Monitor first epoch**: Watch GPU usage during first epoch to see if you can increase batch size

3. **Use mixed precision**: The script already uses `amp=True` for automatic mixed precision training (saves ~30-40% VRAM)

4. **Don't cache images**: Script already sets `cache=False` to avoid loading all images into RAM/VRAM

5. **Save checkpoints**: Script saves every 20 epochs so you can resume if training crashes

6. **Night training**: Start training before bed if it takes many hours

7. **Google Colab**: If your GPU is too weak, use Google Colab's free T4 GPU (15GB VRAM)

---

## 🔄 Transfer Learning Alternative

If training from scratch is too slow or crashes due to low VRAM, use transfer learning instead:

```bash
python train_yolov11x_unified.py --epochs 100
```

Transfer learning:
- ✅ 10-20x faster convergence
- ✅ Uses less GPU memory
- ✅ Better results with small datasets
- ✅ Only needs 50-100 epochs
- ❌ Not optimized specifically for grayscale

---

## 📈 Check Your GPU Specs

```bash
# Check GPU model and VRAM
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"
```

---

## 🎓 Summary

**To lower GPU load, in order of impact:**

1. **Choose smaller model**: `--model-size n` or `--model-size s`
2. **Reduce batch size**: `--batch 1` or `--batch 2`
3. **Reduce image size**: `--imgsz 320` or `--imgsz 416`
4. **Close other programs** using GPU
5. **Clear GPU cache** before training

**Recommended starting point:**
```bash
python train_from_scratch_grayscale.py --model-size n --batch 2 --imgsz 416 --epochs 200
```

This will work on most GPUs with 4GB+ VRAM and provide reasonable accuracy for your specific use case.
