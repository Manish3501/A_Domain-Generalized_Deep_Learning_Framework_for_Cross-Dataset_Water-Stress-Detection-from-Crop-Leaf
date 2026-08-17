#-----------------------------------------------------------------------------------------------------------------------

# Section 1 - Import Libraries
 
import torch
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------

# Section 2 - Basic Training Function

# Trains normal CNN models
# Used for:
# - Tomato Model
# - Maize Model
# - Maize2 Model

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=5
):

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        # Training phase
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            # Move data to CPU/GPU
            images = images.to(device)
            labels = labels.to(device)

            # Clear old gradients - if not reset them then the gardients from previous batch adds to the current one.
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            # Training accuracy
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        avg_train_loss = running_loss / len(train_loader)

        train_accuracy = 100 * correct / total

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)


        # Validation phase

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():   # This tells pytorch not to track gradients during validation which saves memory and time.

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_accuracy = 100 * correct / total

        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        print("-" * 75)

    return (
        train_losses,
        train_accuracies,
        val_accuracies
    )

#-----------------------------------------------------------------------------------------------------------------------

# Section 3 - Feature Fusion Training

def train_fusion_model(
    model,
    fusion_tomato_loader,
    fusion_maize_loader,
    fusion_maize2_loader,
    fusion_tomato_val_loader,
    fusion_maize_val_loader,
    fusion_maize2_val_loader,
    fusion_optimizer,
    fusion_criterion,
    device,
    epochs=10
):

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_weights = None

    for epoch in range(epochs):

        # Training phase
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for (

            (tomato_imgs, tomato_labels),
            (maize_imgs, maize_labels),
            (maize2_imgs, maize2_labels)

        ) in zip(                                     # zip() Iterates three data loaders simultaneously, drawing one batch from each per step. 

            fusion_tomato_loader,
            fusion_maize_loader,
            fusion_maize2_loader
        ):

            tomato_imgs = tomato_imgs.to(device)
            maize_imgs = maize_imgs.to(device)
            maize2_imgs = maize2_imgs.to(device)
            labels = tomato_labels.to(device)

            fusion_optimizer.zero_grad()

            outputs = model(
                tomato_imgs,
                maize_imgs,
                maize2_imgs
            )

            loss = fusion_criterion(
                outputs,
                labels
            )

            loss.backward()

            fusion_optimizer.step()

            running_loss += loss.item()

            # Training accuracy
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        avg_loss = running_loss / len(fusion_tomato_loader)

        train_accuracy = 100 * correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)


        # Validation phase

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for (

                (tomato_imgs, tomato_labels),
                (maize_imgs, maize_labels),
                (maize2_imgs, maize2_labels)

            ) in zip(

                fusion_tomato_val_loader,
                fusion_maize_val_loader,
                fusion_maize2_val_loader
            ):

                tomato_imgs = tomato_imgs.to(device)
                maize_imgs = maize_imgs.to(device)
                maize2_imgs = maize2_imgs.to(device)
                labels = tomato_labels.to(device)

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

        val_accuracy = 100 * correct / total

        val_accuracies.append(val_accuracy)

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_weights = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Fusion Loss: {avg_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )
        print("-" * 75)

    if best_model_weights:

        model.load_state_dict(best_model_weights)

    return (
        train_losses,
        train_accuracies,
        val_accuracies
    )

#-----------------------------------------------------------------------------------------------------------------------

# Section 4 - Domain Generalisation Training

def train_domain_generalisation(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    dg_stress_criterion,
    dg_domain_criterion,
    device,
    epochs=25
):

    train_stress_losses = []
    train_domain_losses = []
    train_accuracies = []
    train_domain_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_weights = None

    for epoch in range(epochs):

        model.train()

        p = epoch / epochs

        lambda_ = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0           # The formula is a sigmoid function that outputs 0.0 at p=0, 0.46 at p=0.5, 0.96 at p=1.0

        model.gradient_reversal.lambda_ = lambda_

        running_stress_loss = 0.0
        running_domain_loss = 0.0

        correct = 0
        total = 0
        domain_correct = 0
        domain_total = 0

        # Training Phase

        for images, stress_labels, domain_labels in train_loader:

            images = images.to(device)
            stress_labels = stress_labels.to(device)
            domain_labels = domain_labels.to(device)

            optimizer.zero_grad()

            stress_out, domain_out = model(images)

            stress_loss = dg_stress_criterion(
                stress_out,
                stress_labels
            )                                             # Cross-entropy between predicted stress class and true stress label. This is the primary objective.

            domain_loss = dg_domain_criterion(
                domain_out,
                domain_labels
            )                                             # Cross-entropy between predicted domain class and true domain label. 
                                                          # Because of the GRL, minimising this loss actually makes the backbone MORE domain-invariant (the GRL flips the gradient direction).
            
            # Track domain classifier accuracy
            _, domain_predicted = torch.max(domain_out, 1)
            domain_correct += (domain_predicted == domain_labels).sum().item()
            domain_total += domain_labels.size(0)

            total_loss = stress_loss + lambda_ * domain_loss                   # weighted sum

            total_loss.backward()

            optimizer.step()

            running_stress_loss += stress_loss.item()

            running_domain_loss += domain_loss.item()

            # Training accuracy
            _, predicted = torch.max(stress_out, 1)

            total += stress_labels.size(0)

            correct += (
                predicted == stress_labels
            ).sum().item()

        avg_stress_loss = (
            running_stress_loss / len(train_loader)
        )

        avg_domain_loss = (
            running_domain_loss / len(train_loader)
        )

        train_accuracy = 100 * correct / total

        domain_accuracy = 100 * domain_correct / domain_total


        train_stress_losses.append(avg_stress_loss)
        train_domain_losses.append(avg_domain_loss)
        train_accuracies.append(train_accuracy)
        train_domain_accuracies.append(domain_accuracy)
        
        # Validation Phase

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, stress_labels, _ in val_loader:

                images = images.to(device)
                stress_labels = stress_labels.to(device)

                stress_out, _ = model(images)

                _, predicted = torch.max(stress_out, 1)

                total += stress_labels.size(0)

                correct += (
                    predicted == stress_labels
                ).sum().item()

        val_accuracy = 100 * correct / total

        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_weights = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }

        scheduler.step(val_accuracy)

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] | "
            f"λ={lambda_:.3f} | "
            f"Stress Loss: {avg_stress_loss:.4f} | "
            f"Domain Loss: {avg_domain_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Domain Acc: {domain_accuracy:.2f}% | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        print("-" * 75)

    """if best_model_weights:

        model.load_state_dict(best_model_weights)

        print(
            f"\nBest Val Accuracy: "
            f"{best_val_accuracy:.2f}% — weights restored."
        )"""

    return (
        train_stress_losses,
        train_domain_losses,
        train_accuracies,
        train_domain_accuracies,
        val_accuracies
    )

#-----------------------------------------------------------------------------------------------------------------------