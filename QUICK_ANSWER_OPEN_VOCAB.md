# Quick Answer: Why Custom Model Can't Be Open-Vocabulary

## The Short Answer

The custom model **has** the open-vocabulary architecture, but **the weights were optimized for specific text prompts** during fine-tuning.

Think of it like this:
- 🏗️ **Architecture** = The building (still the same)
- ⚙️ **Weights** = The wiring and plumbing (completely reconfigured)

The building is the same, but the internal wiring now only responds to specific signals!

---

## What Happens During Fine-Tuning

### Training Data:
```
Image 1: [photo of outlet] + Label: "outlet"
Image 2: [photo of outlet] + Label: "outlet"  
Image 3: [photo of outlet] + Label: "outlet"
...
```

### What The Model Learns:
```
Text embedding for "outlet" 
    ↓
    STRONG connection (optimized by gradients)
    ↓
Visual features of outlets
```

### What The Model DOESN'T Learn:
```
Text embedding for "wall socket" 
    ↓
    NO connection (never in training data)
    ↓
Visual features of outlets
```

---

## The Mathematical Reason

During training, backpropagation optimizes:

```python
# This gets optimized (thousands of gradient updates)
loss = CrossAttention(
    text_embedding("outlet"),      # ← This specific embedding
    visual_features(outlet_image)  # ← These features
)

# This NEVER gets optimized (zero gradient updates)
loss = CrossAttention(
    text_embedding("wall socket"),  # ← Never in training data!
    visual_features(outlet_image)
)
```

The model literally never learned to connect "wall socket" embeddings to outlet visual features.

---

## Analogy: Training a Dog

**Base Model = Dog that knows "sit" means "sit down":**
- You say "sit" → Dog sits ✅
- You say "sit down" → Dog sits ✅  
- You say "take a seat" → Dog sits ✅
- (Dog generalizes to similar commands)

**Custom Model = Dog trained ONLY with "sit" (100 times):**
- You say "sit" → Dog sits immediately! ✅ (highly tuned)
- You say "sit down" → Dog is confused ❌
- You say "take a seat" → Dog is confused ❌
- (Dog only knows the exact command it was trained on)

---

## Can We Fix It?

### YES! You have several options:

### Option 1: Retrain with ALL vocabulary you want (RECOMMENDED)
```python
# During training, use all synonyms
class_names = {
    0: "outlet, wall socket, power outlet, electrical socket",
    1: "fire extinguisher, fire safety equipment",
    2: "fluorescent tube, light fixture, ceiling light"
}
```

### Option 2: Use BOTH models (CURRENT APPROACH) ✅
```python
# What your GUI does now:
if need_flexibility:
    use_base_model_with_synonyms()  # Open-vocabulary
elif need_precision:
    use_custom_model_with_exact_names()  # High precision
```

### Option 3: Freeze text encoder during training
```python
# Only tune vision encoder, keep text encoder general
for param in model.text_encoder.parameters():
    param.requires_grad = False
```

---

## Test It Yourself

Run these commands to see the difference:

```bash
# 1. See the architecture (it's the same!)
python check_model_type.py

# 2. Test with different prompts
python test_custom_model_prompts.py

# 3. Visualize the embedding space
python visualize_embedding_space.py
```

---

## Bottom Line

**Q: Why can't the custom model be open-vocabulary with adjusted weights?**

**A:** Because "adjusted weights" means the model was optimized for specific prompts. The weights are no longer general-purpose - they're specialized. This is exactly what makes it better at detecting the specific objects it was trained on, but also what makes it unable to generalize to synonyms.

It's not a bug, it's a **fundamental trade-off**:
- ⚖️ Specialization ↔ Generalization
- ⚖️ Precision ↔ Flexibility  
- ⚖️ High accuracy on known classes ↔ Works with any vocabulary

You can't have both in the same model (unless you retrain with all the vocabulary you want).

---

## Your Current Solution is Actually Great! 🎯

Your GUI lets users **choose** based on their needs:
- Need to find any socket-like object? → Use base model
- Need precise outlet detection? → Use custom model

This is a **professional, practical solution** that gives users the best of both worlds!
