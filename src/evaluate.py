import torch
import numpy as np
import cv2


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

    print(f"Ensemble Accuracy: {accuracy:.2f}%")

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

    print(f"Improved Ensemble Accuracy: {accuracy:.2f}%")

    return accuracy