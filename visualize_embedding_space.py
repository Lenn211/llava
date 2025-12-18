"""
Visual demonstration of why fine-tuning breaks open-vocabulary detection
Creates a visualization of embedding space before and after fine-tuning
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ============================================================================
# LEFT PLOT: Base Model (Open-Vocabulary)
# ============================================================================
ax1.set_xlim(-1, 11)
ax1.set_ylim(-1, 11)
ax1.set_aspect('equal')
ax1.set_title('Base YOLOv8-World Model\n(Open-Vocabulary Capability)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.axis('off')

# Draw semantic cluster for "socket" concepts
cluster_center = (3, 6)
cluster = Circle(cluster_center, 2.5, color='lightblue', alpha=0.3, ec='blue', linewidth=2)
ax1.add_patch(cluster)
ax1.text(cluster_center[0], cluster_center[1] + 3, 
         'Semantic Cluster:\nSocket-related concepts', 
         ha='center', fontsize=11, fontweight='bold', color='darkblue')

# Plot various synonym embeddings (all close together)
synonyms = [
    ("outlet", (3.0, 6.0)),
    ("wall socket", (2.5, 7.0)),
    ("power outlet", (3.5, 7.0)),
    ("electrical socket", (2.3, 5.2)),
    ("power point", (3.8, 5.5))
]

for label, pos in synonyms:
    ax1.plot(pos[0], pos[1], 'o', markersize=10, color='darkblue')
    ax1.text(pos[0], pos[1] - 0.3, label, ha='center', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.8))

# Draw visual features region
visual_box = FancyBboxPatch((7, 4.5), 3, 3, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='green', facecolor='lightgreen', 
                           alpha=0.3, linewidth=2)
ax1.add_patch(visual_box)
ax1.text(8.5, 7.8, 'Visual Features\n(Actual Socket Image)', 
         ha='center', fontsize=10, fontweight='bold', color='darkgreen')

# Draw arrows from all embeddings to visual features
for label, pos in synonyms:
    ax1.annotate('', xy=(7.5, 6), xytext=pos,
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, alpha=0.5))

# Add detection result
ax1.text(8.5, 3.5, '✅ ALL prompts\ntrigger detection', 
         ha='center', fontsize=11, fontweight='bold', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green'))

# Add legend
ax1.text(0.5, 1, 
         'Cross-attention learns:\nAny socket-related text → Socket visual features',
         fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange'))

# ============================================================================
# RIGHT PLOT: Custom Model (Fine-Tuned)
# ============================================================================
ax2.set_xlim(-1, 11)
ax2.set_ylim(-1, 11)
ax2.set_aspect('equal')
ax2.set_title('Custom YOLOv8x Model\n(After Fine-Tuning on "outlet")', 
              fontsize=14, fontweight='bold', pad=20)
ax2.axis('off')

# Draw concentrated point for "outlet" (trained class)
trained_pos = (3, 6)
trained_cluster = Circle(trained_pos, 0.5, color='red', alpha=0.5, ec='darkred', linewidth=3)
ax2.add_patch(trained_cluster)
ax2.text(trained_pos[0], trained_pos[1] + 1.2, 
         'Trained Class\n(Strong Gradient)', 
         ha='center', fontsize=11, fontweight='bold', color='darkred')

# Plot "outlet" with strong connection
ax2.plot(trained_pos[0], trained_pos[1], 'o', markersize=15, color='darkred')
ax2.text(trained_pos[0], trained_pos[1] - 0.4, 'outlet', ha='center', fontsize=10, 
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', edgecolor='red', linewidth=2))

# Plot synonyms scattered (no training, random positions)
untrained_synonyms = [
    ("wall socket", (1.5, 8.5)),
    ("power outlet", (5.0, 8.2)),
    ("electrical socket", (1.0, 4.0)),
    ("power point", (5.5, 3.8))
]

for label, pos in untrained_synonyms:
    ax2.plot(pos[0], pos[1], 'x', markersize=8, color='gray', mew=2)
    ax2.text(pos[0], pos[1] - 0.3, label, ha='center', fontsize=8, 
             color='gray',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', 
                      edgecolor='gray', alpha=0.5))

# Draw visual features region
visual_box2 = FancyBboxPatch((7, 4.5), 3, 3, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='green', facecolor='lightgreen', 
                            alpha=0.3, linewidth=2)
ax2.add_patch(visual_box2)
ax2.text(8.5, 7.8, 'Visual Features\n(Actual Socket Image)', 
         ha='center', fontsize=10, fontweight='bold', color='darkgreen')

# Strong arrow from "outlet" to visual features
ax2.annotate('', xy=(7.5, 6), xytext=trained_pos,
            arrowprops=dict(arrowstyle='->', color='red', lw=4, alpha=0.8))
ax2.text(5.2, 6.3, 'STRONG\nconnection', ha='center', fontsize=9, 
         fontweight='bold', color='red')

# Weak/broken arrows from synonyms
for label, pos in untrained_synonyms[:2]:  # Only show 2 to avoid clutter
    ax2.annotate('', xy=(7.5, 6), xytext=pos,
                arrowprops=dict(arrowstyle='->', color='gray', 
                              lw=1, alpha=0.3, linestyle='dashed'))

ax2.text(3.5, 9.5, '❌ Weak/No connection\n(Never trained together)', 
         ha='center', fontsize=9, color='gray', style='italic')

# Add detection results
ax2.text(8.5, 3.5, '✅ "outlet" works\n❌ Synonyms fail', 
         ha='center', fontsize=11, fontweight='bold', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange'))

# Add legend
ax2.text(0.5, 1, 
         'Cross-attention learned:\nONLY "outlet" text → Socket visual features\n(No gradient for other terms)',
         fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', edgecolor='red'))

# ============================================================================
# Add overall title and explanation
# ============================================================================
fig.suptitle('Why Fine-Tuning Breaks Open-Vocabulary Detection', 
            fontsize=16, fontweight='bold', y=0.98)

# Add explanation at bottom
explanation = """
KEY INSIGHT: Fine-tuning optimizes the text encoder and cross-attention weights for SPECIFIC prompts.
The model learns a strong connection between "outlet" embedding → socket visual features.
But synonyms ("wall socket", "power outlet", etc.) produce different embeddings that were never
paired with socket images during training, so the cross-attention doesn't activate for them.
"""

fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, 
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', 
                 edgecolor='teal', linewidth=2),
        wrap=True)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('embedding_space_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization to: embedding_space_visualization.png")
print("\nThis diagram shows:")
print("  LEFT: Base model - all synonyms cluster together, all trigger detection")
print("  RIGHT: Custom model - only 'outlet' has strong connection, synonyms don't work")
plt.show()
