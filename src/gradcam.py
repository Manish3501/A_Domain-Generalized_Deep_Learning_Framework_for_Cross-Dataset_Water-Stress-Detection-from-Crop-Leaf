
# Note - Used AI (Claude Sonet 4.6 to derive all below functions for the Grad-CAM implementation. I have modified the code to fit my project and added comments for clarity.

#-----------------------------------------------------------------------------------------------------------------------

# Section 1 - Imports

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


#-----------------------------------------------------------------------------------------------------------------------

# Section 2 - Grad-CAM Class


class StressModelWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        stress_out, _ = self.model(x)
        return stress_out

#-----------------------------------------------------------------------------------------------------------------------


def find_good_examples(
    model,
    data,
    labels,
    transform,
    device,
    target_class,
    n=1,
    min_conf=0.80
):

    model.eval()

    labels = np.array(labels)
    found = []

    candidate_idx = np.where(labels == target_class)[0]

    for idx in candidate_idx:

        image_input = data[idx]

        # If image is a file path
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        # If image is already a numpy array
        else:
            image = image_input.astype("uint8")


        # Same validation transform used for testing
        input_tensor = transform(image)

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            stress_out, _ = model(input_tensor)

            probs = torch.softmax(stress_out, dim=1)

            pred = stress_out.argmax(dim=1).item()

            conf = probs[0, pred].item()


        # Correct + confident
        if (
            pred == target_class
            and conf >= min_conf
        ):
            found.append(idx)

        if len(found) >= n:
            break

    return found

#-----------------------------------------------------------------------------------------------------------------------

def good_pair(
    model,
    data,
    labels,
    transform,
    device,
    min_conf=0.80
):

    stress_idx = find_good_examples(
        model,
        data,
        labels,
        transform,
        device,
        target_class=1,
        n=1,
        min_conf=min_conf
    )

    nonstress_idx = find_good_examples(
        model,
        data,
        labels,
        transform,
        device,
        target_class=0,
        n=1,
        min_conf=min_conf
    )

    pairs = []

    if stress_idx:
        pairs.append(
            (data[stress_idx[0]], "Stress")
        )

    if nonstress_idx:
        pairs.append(
            (data[nonstress_idx[0]], "Non-stress")
        )

    return pairs

#-----------------------------------------------------------------------------------------------------------------------

def generate_gradcam(
    model,
    target_layer,
    domain_images,
    transform,
    device,
    save_path=None
):

    cam_model = StressModelWrapper(model).to(device)
    cam_model.eval()

    grad_cam = GradCAM(
        model=cam_model,
        target_layers=[target_layer]
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device
    )

    domains = list(domain_images.keys())

    fig, axes = plt.subplots(
        len(domains),
        4,
        figsize=(10, 7)
    )

    for row, domain in enumerate(domains):

        for pair, (img_path, label) in enumerate(
            domain_images[domain]
        ):

            # Load original image
            if isinstance(img_path, str):

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            else:

                image = img_path.astype("uint8")

            # Apply the SAME validation transform
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Create display image from transformed tensor
            display_img = input_tensor[0].clone()

            display_img = (
                display_img * std[:, None, None]
                + mean[:, None, None]
            )

            display_img = torch.clamp(
                display_img, 0, 1
            )

            display_img = (
                display_img
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            target_class = (
                1 if label == "Stress" else 0
            )

            targets = [
                ClassifierOutputTarget(target_class)
            ]

            grayscale_cam = grad_cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

            visualization = show_cam_on_image(
                display_img,
                grayscale_cam,
                use_rgb=True
            )

            col_original = pair * 2
            col_gradcam = pair * 2 + 1

            axes[row, col_original].imshow(display_img)
            axes[row, col_original].axis("off")

            axes[row, col_gradcam].imshow(visualization)
            axes[row, col_gradcam].axis("off")

            if row == 0:

                axes[row, col_original].set_title(
                    f"{label}\n(original)",
                    fontsize=10
                )

                axes[row, col_gradcam].set_title(
                    f"{label}\n(Grad-CAM)",
                    fontsize=10
                )

        axes[row, 0].text(
            -0.18,
            0.5,
            domain,
            fontsize=12,
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row, 0].transAxes
        )

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------