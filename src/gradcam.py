#-----------------------------------------------------------------------------------------------------------------------

# Section 1 - Imports

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------

# Section 2 - Grad-CAM Class

class GradCAM:
    """
    Computes a Grad-CAM heatmap for the DANN model's STRESS prediction.

    It hooks into the last convolutional layer of the shared MobileNetV2
    backbone (dg_model.feature_extractor) to capture:
      - activations: what the conv layer outputs (the feature maps)
      - gradients:   how much each feature map influenced the stress score

    The heatmap = ReLU( weighted sum of feature maps ), where the weights are
    the average gradients. Bright regions = most important for the prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register a forward hook to capture the conv layer's output (activations)
        self.target_layer.register_forward_hook(self.save_activations)

    def save_activations(self, module, input, output):
        # Store the feature maps produced by the target layer
        self.activations = output
        # Only register the gradient hook when gradients are actually being
        # tracked. During no_grad forward passes (e.g. just checking a
        # prediction), the tensor has no grad and registering would error.
        if output.requires_grad:
            output.register_hook(self.save_gradients)
            
    def save_gradients(self, grad):
        # Store the gradient flowing into the target layer's activations
        self.gradients = grad.detach()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: a single preprocessed image, shape (1, 3, 224, 224)
        target_class: 0 (Non-stress) or 1 (Stress). If None, uses the
                      model's own prediction.

        Returns:
            cam          - the heatmap as a 2D numpy array, values 0..1
            target_class - the class the heatmap explains
            confidence   - softmax probability for that class
        """
        self.model.eval()

        # Forward pass — DANN returns (stress_out, domain_out)
        # We only care about the stress output for Grad-CAM
        stress_out, _ = self.model(input_tensor)

        # Probability of each class
        probs = F.softmax(stress_out, dim=1)

        # If no target class given, explain the model's own prediction
        if target_class is None:
            target_class = stress_out.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass — compute gradients w.r.t. the target class score
        self.model.zero_grad()
        score = stress_out[0, target_class]
        score.backward(retain_graph=True)

        # weights = global average pool of gradients over the spatial dimensions
        # shape: (1, channels, 1, 1) — one weight per feature map
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of the feature maps, then ReLU
        # (ReLU keeps only features that POSITIVELY influence the class)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Convert to numpy and normalise to 0..1 for display
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class, confidence

#-----------------------------------------------------------------------------------------------------------------------

# Section 3 - Image Loading Helper

def load_image_for_gradcam(image_input, transform, device):
    """
    Loads and preprocesses a single image, matching the LeafDataset pipeline.

    image_input: file path (str) OR numpy array (H, W, C) in RGB
    transform:   use dg_val_transform (no augmentation, for inference)

    Returns:
        input_tensor - shape (1, 3, 224, 224), ready for the model
        display_img  - the original image as a 0..1 RGB numpy array for plotting
    """
    # Load exactly like LeafDataset.__getitem__
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_input.astype("uint8")

    # Keep a resized copy for display (before normalisation)
    display_img = cv2.resize(image, (224, 224)) / 255.0

    # Apply the same transform used at validation/test time
    input_tensor = transform(image).unsqueeze(0).to(device)

    return input_tensor, display_img

#-----------------------------------------------------------------------------------------------------------------------

# Section 4 - Overlay Helper

def overlay_heatmap(cam, display_img, alpha=0.5):
    """
    Overlays the Grad-CAM heatmap on top of the original image.

    cam:         2D heatmap (0..1) from GradCAM.generate
    display_img: original image (0..1 RGB)
    alpha:       blend weight (0.5 = equal mix)

    Returns:
        overlay - blended RGB image (0..1) for plotting
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(cam, (display_img.shape[1], display_img.shape[0]))

    # Apply the JET colourmap (blue = low attention, red = high attention)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    # Blend heatmap with original image
    overlay = alpha * heatmap + (1 - alpha) * display_img
    overlay = np.clip(overlay, 0, 1)

    return overlay

#-----------------------------------------------------------------------------------------------------------------------

# Section 5 - Paper-Style Grid (multiple conditions per row)

def gradcam_grid(
    model,
    target_layer,
    images,            # list of (image_input, column_label) tuples
    transform,
    device,
    row_title="Grad-CAM (DG model)",
    save_path=None
):
    """
    Produces a 2-row grid like the reference paper:
      Row 1 — original leaf images
      Row 2 — Grad-CAM overlays showing where the model looked

    images: a list like
        [(path_or_array, "25% capacity"),
         (path_or_array, "50% capacity"),
         (path_or_array, "75% capacity"),
         (path_or_array, "100% capacity")]
      Each tuple becomes one column. The label is the column header.

    save_path: if given, saves the figure as a PNG (e.g. for your report)
    """

    gradcam = GradCAM(model, target_layer)

    n = len(images)
    label_names = {0: "Non-stress", 1: "Stress"}

    # 2 rows (original, overlay) × n columns
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.8))

    # Handle the case of a single column (axes would be 1D)
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (img_input, col_label) in enumerate(images):

        # Load and preprocess
        input_tensor, display_img = load_image_for_gradcam(
            img_input, transform, device
        )

        # Generate Grad-CAM for the model's own prediction
        cam, pred_class, conf = gradcam.generate(input_tensor)

        # Build overlay
        overlay = overlay_heatmap(cam, display_img)

        # --- Row 1: original leaf image ---
        axes[0, col].imshow(display_img)
        axes[0, col].set_title(col_label, fontsize=13)
        axes[0, col].axis("off")

        # --- Row 2: Grad-CAM overlay ---
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(
            f"{label_names[pred_class]}  ({conf*100:.0f}%)",
            fontsize=11,
            color=("#0F6E56" if pred_class == 1 else "#993C1D")
        )
        axes[1, col].axis("off")

    # Row labels on the left
    axes[0, 0].set_ylabel("Leaf image", fontsize=13, rotation=90)
    axes[1, 0].set_ylabel(row_title, fontsize=13, rotation=90)
    # Re-enable y-axis label visibility (axis off hides it, so use text instead)
    axes[0, 0].text(-0.12, 0.5, "Leaf image", fontsize=13, rotation=90,
                    va="center", ha="center", transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.12, 0.5, row_title, fontsize=13, rotation=90,
                    va="center", ha="center", transform=axes[1, 0].transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved Grad-CAM grid to {save_path}")

    plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# Section 6 - Multi-Domain Grid (proves generalisation across plant types)

def gradcam_multidomain_grid(
    model,
    target_layer,
    domain_images,     # dict: {"Tomato": [(img, label), ...], "Maize": [...], ...}
    transform,
    device,
    save_path=None
):
    """
    Produces a grid with ONE ROW PER DOMAIN, showing Grad-CAM overlays across
    different stress levels in each domain.

    This is the strongest visual evidence for Domain Generalisation: the model
    should focus on leaf tissue (not background) consistently across all three
    visually-different domains.

    domain_images: dict mapping domain name -> list of (image, column_label)
        {
          "Tomato":  [(t1, "Stress"), (t2, "Non-stress")],
          "Maize":   [(m1, "Stress"), (m2, "Non-stress")],
          "Maize2":  [(n1, "Stress"), (n2, "Non-stress")],
        }
      All rows must have the same number of columns.
    """

    gradcam = GradCAM(model, target_layer)
    label_names = {0: "Non-stress", 1: "Stress"}

    domains = list(domain_images.keys())
    n_rows = len(domains)
    n_cols = len(domain_images[domains[0]])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, n_cols)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for row, domain in enumerate(domains):
        for col, (img_input, col_label) in enumerate(domain_images[domain]):

            input_tensor, display_img = load_image_for_gradcam(
                img_input, transform, device
            )
            cam, pred_class, conf = gradcam.generate(input_tensor)
            overlay = overlay_heatmap(cam, display_img)

            axes[row, col].imshow(overlay)
            axes[row, col].axis("off")

            # Column header only on the top row
            if row == 0:
                axes[row, col].set_title(col_label, fontsize=12)

            # Show prediction below each
            axes[row, col].text(
                0.5, -0.08,
                f"{label_names[pred_class]} ({conf*100:.0f}%)",
                fontsize=10, ha="center", va="top",
                transform=axes[row, col].transAxes,
                color=("#0F6E56" if pred_class == 1 else "#993C1D")
            )

        # Row label = domain name
        axes[row, 0].text(
            -0.15, 0.5, domain, fontsize=13, rotation=90,
            va="center", ha="center", transform=axes[row, 0].transAxes
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved multi-domain Grad-CAM grid to {save_path}")

    plt.show()

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------

# Section 7 - Find Correctly-Predicted, Confident Examples

def find_good_examples(model, data, labels, transform, device,
                       target_class, n=1, min_conf=0.80):
    """
    Returns indices of images that the model predicts CORRECTLY and with high
    confidence, for a given class. This avoids putting wrong predictions in
    your paper figure.

    target_class: 0 (Non-stress) or 1 (Stress)
    n:            how many examples to return
    min_conf:     minimum softmax confidence (0.80 = 80%)
    """
    model.eval()
    labels_arr = np.array(labels)
    found = []

    # Only look at images whose TRUE label matches the class we want
    candidate_idx = np.where(labels_arr == target_class)[0]

    for idx in candidate_idx:
        # Load and preprocess this image
        input_tensor, _ = load_image_for_gradcam(data[idx], transform, device)

        with torch.no_grad():
            stress_out, _ = model(input_tensor)
            probs = F.softmax(stress_out, dim=1)
            pred = stress_out.argmax(dim=1).item()
            conf = probs[0, pred].item()

        # Keep it only if prediction is correct AND confident
        if pred == target_class and conf >= min_conf:
            found.append(idx)

        if len(found) >= n:
            break

    if len(found) < n:
        print(f"  ⚠ Only found {len(found)} good example(s) for class "
              f"{target_class} (wanted {n}). Try lowering min_conf.")

    return found

#-----------------------------------------------------------------------------------------------------------------------

# Section 8 - Multi-Domain Grid WITH Original Image Beside Each Overlay

def gradcam_multidomain_with_original(
    model,
    target_layer,
    domain_images,     # dict: {"Tomato": [(img, label), ...], ...}
    transform,
    device,
    save_path=None
):
    """
    Like gradcam_multidomain_grid, but shows the ORIGINAL image and the
    Grad-CAM overlay side by side for each example.

    Layout per domain row:
        [ original | overlay ]   [ original | overlay ]
           Stress example           Non-stress example
    """

    gradcam = GradCAM(model, target_layer)
    label_names = {0: "Non-stress", 1: "Stress"}

    domains = list(domain_images.keys())
    n_rows = len(domains)
    n_pairs = len(domain_images[domains[0]])   # examples per domain
    n_cols = n_pairs * 2                        # each example = original + overlay

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.0 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    for row, domain in enumerate(domains):
        for pair, (img_input, col_label) in enumerate(domain_images[domain]):

            input_tensor, display_img = load_image_for_gradcam(
                img_input, transform, device
            )
            cam, pred_class, conf = gradcam.generate(input_tensor)
            overlay = overlay_heatmap(cam, display_img)

            col_orig = pair * 2          # original goes in even column
            col_over = pair * 2 + 1      # overlay goes in odd column

            # Original
            axes[row, col_orig].imshow(display_img)
            axes[row, col_orig].axis("off")

            # Overlay
            axes[row, col_over].imshow(overlay)
            axes[row, col_over].axis("off")

            # Column headers only on the top row
            if row == 0:
                axes[row, col_orig].set_title(f"{col_label}\n(original)", fontsize=11)
                axes[row, col_over].set_title(f"{col_label}\n(Grad-CAM)", fontsize=11)

            # Prediction text under the overlay
            axes[row, col_over].text(
                0.5, -0.10,
                f"{label_names[pred_class]} ({conf*100:.0f}%)",
                fontsize=10, ha="center", va="top",
                transform=axes[row, col_over].transAxes,
                color=("#0F6E56" if pred_class == 1 else "#993C1D")
            )

        # Domain name on the left
        axes[row, 0].text(
            -0.18, 0.5, domain, fontsize=13, rotation=90,
            va="center", ha="center", transform=axes[row, 0].transAxes
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved Grad-CAM grid to {save_path}")
    plt.show()