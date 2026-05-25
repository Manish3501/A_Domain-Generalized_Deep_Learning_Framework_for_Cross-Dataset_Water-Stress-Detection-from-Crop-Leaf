import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# SECTION 1 - EVALUATE INDIVIDUAL MODEL
# ============================================================

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

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy


# ============================================================
# SECTION 2 - ENSEMBLE PREDICTION
# ============================================================

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

        pred1 = torch.softmax(
            tomato_model(image_tensor),
            dim=1
        )

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


# ============================================================
# SECTION 3 - ENSEMBLE EVALUATION
# ============================================================

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

    print(f"Ensemble Model Accuracy on Merged Test Dataset: {accuracy:.2f}%")

    return accuracy


# ============================================================
# SECTION 4 - IMPROVED ENSEMBLE PREDICTION
# ============================================================

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


# ============================================================
# SECTION 5 - IMPROVED ENSEMBLE EVALUATION
# ============================================================

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

    print(f"Improved Ensemble Model Accuracy on Merged Test Dataset: {accuracy:.2f}%")

    return accuracy


# ============================================================
# SECTION 6 - FEATURE FUSION EVALUATION
# ============================================================

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

    print(f"Feature Fusion Accuracy: {accuracy:.2f}%")

    return accuracy



# ============================================================
# SECTION 7 - DOMAIN GENERALISATION EVALUATION
# ============================================================

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

    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    accuracy = 100 * correct / total

    print("=" * 50)

    print(
        f"  {dataset_name} Test Accuracy: "
        f"{accuracy:.2f}%"
    )

    print("=" * 50)

    # ========================================================
    # PER-DOMAIN ACCURACY
    # ========================================================

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


# ============================================================
# SECTION 8 - METRIC UTILITIES
# ============================================================

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    classification_report
)


def compute_all_metrics(
    predictions,
    labels,
    probabilities,
    model_name="Model"
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions
    )

    recall = recall_score(
        labels,
        predictions
    )

    f1 = f1_score(
        labels,
        predictions
    )

    auc = roc_auc_score(
        labels,
        probabilities
    )

    print("\n" + "=" * 60)

    print(f"{model_name}")

    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"AUC Score: {auc:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            labels,
            predictions
        )
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "auc": auc
    }


# ============================================================
# SECTION 9 - CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    predictions,
    labels,
    title="Confusion Matrix"
):

    cm = confusion_matrix(
        labels,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(title)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()