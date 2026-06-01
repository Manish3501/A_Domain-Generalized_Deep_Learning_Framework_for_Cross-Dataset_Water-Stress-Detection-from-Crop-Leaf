#-----------------------------------------------------------------------------------------------------------------------

# Section 1 - Import Libraries
 
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

#-----------------------------------------------------------------------------------------------------------------------

# Section 2 - Evaluate Individual Model

def evaluate_model(
    model,
    test_loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            # Move tensors to device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)                 # The batch size (number of images in this batch, typically 32)

            correct += (
                predicted == labels                 # boolean tensor, True where prediction matches label
            ).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy

#-----------------------------------------------------------------------------------------------------------------------

# Section 3 - Ensemble Prediction

def ensemble_predict(
    image_tensor,
    tomato_model,
    maize_model,
    maize2_model,
    device
):

    tomato_model.eval()
    maize_model.eval()
    maize2_model.eval()

    with torch.no_grad():

        image_tensor = image_tensor.unsqueeze(0).to(device)

        # unsqueeze(0) — Adds a batch dimension. A single image tensor has shape (3, 224, 224) — 3 colour channels, 224×224 pixels. 
        # Neural networks always expect batches, so shape must be (1, 3, 224, 224). .unsqueeze(0) inserts a dimension of size 1 at position 0.

        pred1 = torch.softmax(
            tomato_model(image_tensor),
            dim=1
        )                                             # Softmax converts raw model outputs into probabilities and softmax must be apply before averaging.

        pred2 = torch.softmax(
            maize_model(image_tensor),
            dim=1
        )

        pred3 = torch.softmax(
            maize2_model(image_tensor),
            dim=1
        )

        # Average predictions
        final_pred = (
            pred1 + pred2 + pred3
        ) / 3

        predicted_class = torch.argmax(
            final_pred,
            dim=1
        )

    return predicted_class.item()

#-----------------------------------------------------------------------------------------------------------------------

# Section 4 - Ensemble Evaluation

def evaluate_ensemble(
    merged_test_data,
    tomato_model,
    maize_model,
    maize2_model,
    device
):

    correct = 0
    total = 0

    for image, label in merged_test_data:

        prediction = ensemble_predict(
            image,
            tomato_model,
            maize_model,
            maize2_model,
            device
        )

        total += 1

        if prediction == label:

            correct += 1

    accuracy = 100 * correct / total

    print(f"Ensemble Model Accuracy : {accuracy:.2f}%")

    return accuracy

#-----------------------------------------------------------------------------------------------------------------------

# Section 5 - Improved Ensemble Prediction

def improved_ensemble_predict(
    image_tensor,
    improved_tomato_model,
    improved_maize_model,
    improved_maize2_model,
    device
):

    improved_tomato_model.eval()
    improved_maize_model.eval()
    improved_maize2_model.eval()

    with torch.no_grad():

        image_tensor = image_tensor.unsqueeze(0).to(device)

        pred1 = torch.softmax(
            improved_tomato_model(image_tensor),
            dim=1
        )

        pred2 = torch.softmax(
            improved_maize_model(image_tensor),
            dim=1
        )

        pred3 = torch.softmax(
            improved_maize2_model(image_tensor),
            dim=1
        )

        # Weighted averaging
        final_pred = (
            0.5 * pred1 +
            0.3 * pred2 +
            0.2 * pred3
        )

        predicted_class = torch.argmax(
            final_pred,
            dim=1
        )

    return predicted_class.item()

#-----------------------------------------------------------------------------------------------------------------------

# Section 6 - Improved Ensemble Evaluation

def evaluate_improved_ensemble(
    merged_test_data,
    improved_tomato_model,
    improved_maize_model,
    improved_maize2_model,
    device
):

    correct = 0
    total = 0

    for image, label in merged_test_data:

        prediction = improved_ensemble_predict(
            image,
            improved_tomato_model,
            improved_maize_model,
            improved_maize2_model,
            device
        )

        total += 1

        if prediction == label:

            correct += 1

    accuracy = 100 * correct / total

    print(f"Improved Ensemble Model Accuracy : {accuracy:.2f}%")

    return accuracy

#-----------------------------------------------------------------------------------------------------------------------

# Section 7 - Feature Fusion Evaluation

def evaluate_fusion_model(
    model,
    fusion_tomato_test_loader,
    fusion_maize_test_loader,
    fusion_maize2_test_loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for (

            (tomato_imgs, tomato_labels),
            (maize_imgs, maize_labels),
            (maize2_imgs, maize2_labels)

        ) in zip(

            fusion_tomato_test_loader,
            fusion_maize_test_loader,
            fusion_maize2_test_loader
        ):

            # Move to device
            tomato_imgs = tomato_imgs.to(device)
            maize_imgs = maize_imgs.to(device)
            maize2_imgs = maize2_imgs.to(device)
            labels = tomato_labels.to(device)

            # Forward pass
            outputs = model(
                tomato_imgs,
                maize_imgs,
                maize2_imgs
            )

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    print(f"Feature Fusion Model Accuracy: {accuracy:.2f}%")

    return accuracy

#-----------------------------------------------------------------------------------------------------------------------

# Section 8 - Domain Generalisation Evaluation

def evaluate_dg_model(
    model,
    test_loader,
    device,
    dataset_name="Domain Generalisation"
):

    model.eval()

    correct = 0
    total = 0

    # Per-domain tracking
    domain_correct = {
        0: 0,
        1: 0,
        2: 0
    }

    domain_total = {
        0: 0,
        1: 0,
        2: 0
    }

    all_predictions = []
    all_labels = []
    all_domains = []

    with torch.no_grad():

        for images, labels, domains in test_loader:

            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            domains = domains.to(device)

            # Forward pass
            stress_out, domain_out = model(images)

            _, predicted = torch.max(stress_out, 1)

            # Only stress_out is used at evaluation. domain_out is discarded. 
            # At deployment time, the domain ID of an input image is unknown and irrelevant -
            # the model produces one stress prediction regardless of which plant species it is looking at. This is the key practical advantage over the Fusion model.

            # Overall accuracy
            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            # Per-domain accuracy
            for i in range(labels.size(0)):
                domain_id = domains[i].item()
                domain_total[domain_id] += 1

                if predicted[i] == labels[i]:
                    domain_correct[domain_id] += 1

            # Store results
            all_predictions.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_domains.extend(
                domains.cpu().numpy()
            )

    # Overall accuracy

    accuracy = 100 * correct / total

    print("=" * 50)
    print(
        f"  {dataset_name} Test Accuracy: "
        f"{accuracy:.2f}%"
    )
    print("=" * 50)

    # Per-domain accuracy

    domain_names = {
        0: "Tomato",
        1: "Maize (folder)",
        2: "Maize2 (.npy)"
    }

    print("\nPer-Domain Accuracy:")

    for domain_id in domain_total:

        domain_acc = (
            100 * domain_correct[domain_id]
            / domain_total[domain_id]
        )

        print(
            f"  {domain_names[domain_id]:20}: "
            f"{domain_acc:.2f}%  "
            f"({domain_correct[domain_id]}/"
            f"{domain_total[domain_id]})"
        )

    return (
        accuracy,
        all_predictions,
        all_labels,
        all_domains
    )

#-----------------------------------------------------------------------------------------------------------------------

# Section 10 - Collect Predictions from any model

def collect_predictions_single(model, test_loader, device):
    """
    For models that take ONE image and return ONE output (Ensemble, DG models).
    Returns arrays of predictions, true labels, and stress probabilities.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            # Handle both LeafDataset (2 values) and DomainLeafDataset (3 values)
            if len(batch) == 3:
                images, stress_labels, _ = batch   # DG loader — ignore domain_id
            else:
                images, stress_labels = batch      # Regular loader

            images        = images.to(device)
            stress_labels = stress_labels.to(device)

            # For DG model, forward returns (stress_out, domain_out)
            # For regular models, forward returns just stress_out
            output = model(images)
            if isinstance(output, tuple):
                stress_out = output[0]             # Take only stress output from DG
            else:
                stress_out = output

            probs       = torch.softmax(stress_out, dim=1)
            stress_prob = probs[:, 1]              # Probability for class 1 (Stress)
            _, predicted = torch.max(stress_out, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(stress_labels.cpu().numpy())
            all_probs.extend(stress_prob.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )

#-----------------------------------------------------------------------------------------------------------------------

def collect_predictions_fusion(fusion_model, t_loader, m_loader, m2_loader, device):
    """
    For the FusionModel only — needs 3 images simultaneously.
    Uses tomato labels as the ground truth (matching your original fusion code).
    
    NOTE: This is the same design flaw discussed earlier — using only tomato
    labels — but we keep it consistent with your original fusion code so the
    comparison is honest.
    """
    fusion_model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for (t_imgs, t_labels), (m_imgs, _), (m2_imgs, _) in zip(
            t_loader, m_loader, m2_loader
        ):
            t_imgs  = t_imgs.to(device)
            m_imgs  = m_imgs.to(device)
            m2_imgs = m2_imgs.to(device)
            labels  = t_labels.to(device)

            outputs = fusion_model(t_imgs, m_imgs, m2_imgs)
            probs   = torch.softmax(outputs, dim=1)
            stress_prob = probs[:, 1]
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(stress_prob.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )

#-----------------------------------------------------------------------------------------------------------------------

# Section 11 - Define all metrics 

def compute_all_metrics(preds, labels, probs, model_name):
    """
    Computes and prints the complete evaluation suite.
    Returns a dict of all metric values (useful for the comparison table later).
    """
    acc       = 100 * accuracy_score(labels, preds)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall    = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1        = f1_score(labels, preds, pos_label=1, zero_division=0)
    auc       = roc_auc_score(labels, probs)

    print(f"\n{'-'*75}")
    print(f"  {model_name}")
    print(f"{'-'*75}")
    print(f"\n Test Accuracy: {acc:.2f}%")
    print(f"  Precision     : {precision:.4f}  → Of all predicted Stress, how many were really Stress?")
    print(f"  Recall        : {recall:.4f}  → Of all actual Stress plants, how many did model catch?")
    print(f"  F1-Score      : {f1:.4f}  → Balance of Precision and Recall")
    print(f"  ROC-AUC       : {auc:.4f}  → Overall separability (1.0 = perfect, 0.5 = random)")
    print()
    print(classification_report(
        labels, preds,
        target_names=["Non-Stress (0)", "Stress (1)"],
        digits=4
    ))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

#-----------------------------------------------------------------------------------------------------------------------

# Section 10 - Confusion Matrix

def plot_confusion_matrix(preds, labels, model_name):
    """
    Plots two confusion matrices side by side:
      Left  — raw counts (how many images in each cell)
      Right — row-normalised percentages (easier to read when class sizes differ)

    Reading the confusion matrix:
    
              Predicted →    Non-Stress    Stress
    Actual ↓
    Non-Stress               TN            FP   ← False alarm (annoying but safe)
    Stress                   FN            TP   ← MISSED stress (dangerous for agriculture)

    What to tell your mentor:
    - High diagonal = model is correct
    - FN (bottom-left) = stressed plants the model MISSED — most dangerous cell
    - FP (top-right) = non-stressed plants called stress — false alarm
    - For plant stress: you want FN as LOW as possible (high Recall)
    """
    cm     = confusion_matrix(labels, preds)
    cm_pct = confusion_matrix(labels, preds, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=13, y=1.01)

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-Stress", "Stress"],
                yticklabels=["Non-Stress", "Stress"],
                ax=axes[0], annot_kws={"size": 14})
    axes[0].set_title("Raw Counts")
    axes[0].set_ylabel("Actual Label", fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)

    # Normalised percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Greens',
                xticklabels=["Non-Stress", "Stress"],
                yticklabels=["Non-Stress", "Stress"],
                ax=axes[1], annot_kws={"size": 14})
    axes[1].set_title("Row-Normalised (%)")
    axes[1].set_ylabel("Actual Label", fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  → Missed {fn} stressed plants (False Negatives)")
    print(f"  → False alarms on {fp} non-stressed plants (False Positives)")

#-----------------------------------------------------------------------------------------------------------------------

# Section 11 - ROC Curve

def plot_roc_curves(results_list):
    """
    Plots all models' ROC curves on the same axes for easy visual comparison.
    
    What the ROC curve shows:
      X-axis: False Positive Rate = FP / (FP + TN)  →  how many non-stressed get called stressed
      Y-axis: True Positive Rate  = TP / (TP + FN)  →  how many stressed get correctly caught
    
    A perfect model hugs the top-left corner (high TPR, low FPR).
    The diagonal line = random guessing (AUC = 0.5).
    The area under the curve (AUC) summarises the entire curve in one number.
    
    Why ROC is better than accuracy alone:
      It shows performance at EVERY possible threshold, not just the default 0.5.
      You can choose a lower threshold to catch more stressed plants (higher recall)
      at the cost of more false alarms — this trade-off is visible on the ROC curve.
    """
    plt.figure(figsize=(9, 7))
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for i, r in enumerate(results_list):
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        auc = r["auc"]
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 linewidth=2.5, label=f"{r['name']}  (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate  (Non-Stress wrongly called Stress)", fontsize=12)
    plt.ylabel("True Positive Rate  (Stress correctly detected)", fontsize=12)
    plt.title("ROC Curves — All Models Compared", fontsize=14)
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# Section 12 - Overfitting Checker

def check_for_overfitting(train_accuracy, val_accuracy, test_accuracy, model_name):
    """
    Compares training vs validation vs test accuracy to diagnose overfitting.
    
    OVERFITTING:  Train accuracy >> Val/Test accuracy
                  Model memorised training images, fails on new ones.
                  Fix: more augmentation, more dropout, fewer epochs.

    UNDERFITTING: All three are low
                  Model hasn't learned enough.
                  Fix: more epochs, lower learning rate, unfreeze more layers.

    GOOD FIT:     Train ≈ Val ≈ Test (within ~5%)
                  Model generalises well.

    100% TRAIN ACCURACY WARNING:
                  Almost always means overfitting or data leakage.
                  Genuine 100% is only possible on tiny, very easy datasets.
    """
    gap_train_val  = abs(train_accuracy - val_accuracy)
    gap_train_test = abs(train_accuracy - test_accuracy)

    print(f"\n{'─'*75}")
    print(f"  Overfitting Check — {model_name}")
    print(f"{'─'*75}")
    print(f"  Train Accuracy : {train_accuracy:.2f}%")
    print(f"  Val   Accuracy : {val_accuracy:.2f}%")
    print(f"  Test  Accuracy : {test_accuracy:.2f}%")
    print(f"  Train–Val Gap  : {gap_train_val:.2f}%")

    if train_accuracy >= 99.5:
        print("WARNING: Near-100% train accuracy — likely overfitting or data leakage.")
    elif gap_train_val > 15:
        print("WARNING: Large train-val gap — overfitting. Add more dropout/augmentation.")
    elif gap_train_val > 8:
        print("MILD overfitting — acceptable but worth monitoring.")
    else:
        print("Good fit — train and val accuracy are close.")

    if test_accuracy > train_accuracy + 2:
        print("WARNING: Test accuracy > Train accuracy — possible data leakage or very easy test set.")

#-----------------------------------------------------------------------------------------------------------------------


