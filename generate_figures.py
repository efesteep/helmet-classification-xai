"""
generate_figures.py
-------------------
Generates the training/validation curves and the class-distribution figure
used in the final report.

The training histories below were captured on Kaggle (Tesla T4) while training
the three models described in the paper:
  - Custom CNN        : 12 epochs, single phase
  - EfficientNetV2-B0 : 6 epochs frozen backbone + 5 epochs fine-tuning
  - ConvNeXt-Tiny     : 5 epochs frozen backbone + 4 epochs fine-tuning

Usage:
    python generate_figures.py
Outputs (written to ./figures/):
    training_accuracy_curves.png
    training_loss_curves.png
    class_distribution.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
DPI = 300  # report requires >= 300 ppi

# ---------------------------------------------------------------------
# Training histories captured during training
# ---------------------------------------------------------------------
custom_cnn = {
    "accuracy":     [0.7654, 0.8369, 0.8612, 0.8769, 0.8854, 0.8893,
                     0.9028, 0.9029, 0.9046, 0.9116, 0.9133, 0.9131],
    "val_accuracy": [0.7723, 0.8865, 0.8530, 0.8916, 0.8967, 0.9214,
                     0.8356, 0.8941, 0.7843, 0.9232, 0.9131, 0.7679],
    "loss":         [0.4814, 0.3735, 0.3336, 0.2991, 0.2799, 0.2709,
                     0.2502, 0.2416, 0.2370, 0.2328, 0.2220, 0.2209],
    "val_loss":     [0.4726, 0.2921, 0.3571, 0.2722, 0.2580, 0.2202,
                     0.4392, 0.2715, 0.6068, 0.1987, 0.2226, 0.5991],
}

# Phase 1 (frozen backbone) followed by Phase 2 (fine-tuning), concatenated
efficientnet_v2 = {
    "accuracy":     [0.9089, 0.9354, 0.9410, 0.9456, 0.9494, 0.9511,
                     0.9148, 0.9428, 0.9502, 0.9541, 0.9605],
    "val_accuracy": [0.9396, 0.9509, 0.9338, 0.9480, 0.9589, 0.9673,
                     0.9440, 0.9458, 0.9578, 0.9534, 0.9611],
    "loss":         [0.2210, 0.1663, 0.1516, 0.1414, 0.1368, 0.1312,
                     0.1726, 0.1340, 0.1263, 0.1152, 0.1011],
    "val_loss":     [0.1574, 0.1352, 0.1746, 0.1406, 0.1182, 0.1060,
                     0.1569, 0.1549, 0.1242, 0.1360, 0.1193],
}

convnext_tiny = {
    "accuracy":     [0.9038, 0.9395, 0.9471, 0.9505, 0.9520,
                     0.9616, 0.9676, 0.9712, 0.9740],
    "val_accuracy": [0.9523, 0.9578, 0.9596, 0.9633, 0.9614,
                     0.9654, 0.9676, 0.9687, 0.9690],
    "loss":         [0.2274, 0.1559, 0.1379, 0.1210, 0.1233,
                     0.1006, 0.0868, 0.0742, 0.0700],
    "val_loss":     [0.1365, 0.1208, 0.1168, 0.1054, 0.1158,
                     0.1000, 0.0990, 0.0905, 0.0890],
}

MODELS = {
    "Custom CNN":        custom_cnn,
    "EfficientNetV2-B0": efficientnet_v2,
    "ConvNeXt-Tiny":     convnext_tiny,
}
COLORS = {
    "Custom CNN":        "#1f77b4",
    "EfficientNetV2-B0": "#ff7f0e",
    "ConvNeXt-Tiny":     "#2ca02c",
}

# Dataset composition after bounding-box cropping
CLASS_COUNTS = {"helmet": 14305, "no_helmet": 4017}


def plot_curves(metric, val_metric, ylabel, title, filename):
    """Plot train/validation curves for all three models on one figure."""
    plt.figure(figsize=(8, 5))
    for name, history in MODELS.items():
        epochs = range(1, len(history[metric]) + 1)
        color = COLORS[name]
        plt.plot(epochs, history[metric], color=color, label=f"{name} (train)")
        plt.plot(epochs, history[val_metric], "--", color=color,
                 label=f"{name} (val)")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_class_distribution():
    """Bar chart of the class distribution after cropping."""
    plt.figure(figsize=(6, 5))
    bars = plt.bar(list(CLASS_COUNTS.keys()), list(CLASS_COUNTS.values()),
                   color=["#2c7fb8", "#de2d26"])
    top = max(CLASS_COUNTS.values())
    for bar, value in zip(bars, CLASS_COUNTS.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, value + top * 0.01,
                 f"{value:,}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    plt.ylabel("Number of cropped images")
    plt.title("Class Distribution After Bounding-Box Cropping")
    plt.tight_layout()
    path = os.path.join(OUT, "class_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_curves("accuracy", "val_accuracy", "Accuracy",
                "Training and Validation Accuracy",
                "training_accuracy_curves.png")
    plot_curves("loss", "val_loss", "Loss",
                "Training and Validation Loss",
                "training_loss_curves.png")
    plot_class_distribution()
    print("All figures generated.")
