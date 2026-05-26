#-----------------------------------------------------------------------------------------------------------------------

# Section 1 - Import Libraries
 
import torch
import torch.nn as nn
from torchvision import models

#-----------------------------------------------------------------------------------------------------------------------

# Section 2 - Device Configuration

# Apple Silicon GPU (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")

# NVIDIA GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")

# CPU
else:
    device = torch.device("cpu")
    print("Using CPU")

#-----------------------------------------------------------------------------------------------------------------------

# Section 3 - Create Base Model

# The base model I have used is moilenetv2 which is a lightweight CNN architecture that is efficient and performs well on image classification tasks. 
# I have replaced the final layer to output 2 classes (stress vs non-stress) and moved the model for training.
# 0 = Non-stress
# 1 = Stress

def create_model():

    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    # Replace final layer for binary classification
    model.classifier[1] = nn.Linear(
        model.last_channel,
        2
    )

    # Move model to GPU/CPU
    model = model.to(device)

    return model

#-----------------------------------------------------------------------------------------------------------------------

# Section 4 - Feature Extractor

# This is used for feature fusion approach where we want to extract deep features from each dataset and then combine them for final classification.
# I removed classifier and kept only deep feature extraction.

def create_feature_extractor():

    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    # Remove classifier
    feature_extractor = model.features

    # Move to device
    feature_extractor = feature_extractor.to(device)

    return feature_extractor

#-----------------------------------------------------------------------------------------------------------------------

# Section 5 - Create Feature Extractors

tomato_feature_extractor = create_feature_extractor()
maize_feature_extractor = create_feature_extractor()
maize2_feature_extractor = create_feature_extractor()

print("Feature Extractors Created")

#-----------------------------------------------------------------------------------------------------------------------

# Section 6 - Feature Fusion Model

# Combines features from:
# - Tomato dataset
# - Maize dataset
# - Maize2 dataset

class FusionModel(nn.Module):

    def __init__(self):

        super(FusionModel, self).__init__()

        # Feature extractors
        self.tomato_extractor = tomato_feature_extractor
        self.maize_extractor = maize_feature_extractor
        self.maize2_extractor = maize2_feature_extractor

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Final classifier
        self.classifier = nn.Sequential(

            nn.Linear(1280 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(
        self,
        tomato_img,
        maize_img,
        maize2_img
    ):

        # Extract features
        tomato_feat = self.tomato_extractor(tomato_img)
        maize_feat = self.maize_extractor(maize_img)
        maize2_feat = self.maize2_extractor(maize2_img)

        # Global pooling
        tomato_feat = self.pool(tomato_feat)
        maize_feat = self.pool(maize_feat)
        maize2_feat = self.pool(maize2_feat)

        # Flatten
        tomato_feat = torch.flatten(tomato_feat, 1)
        maize_feat = torch.flatten(maize_feat, 1)
        maize2_feat = torch.flatten(maize2_feat, 1)

        # Feature fusion
        fused_features = torch.cat(
            [tomato_feat, maize_feat, maize2_feat],
            dim=1
        )

        # Final prediction
        output = self.classifier(fused_features)

        return output

#-----------------------------------------------------------------------------------------------------------------------

# Section 7 - Domain Generalization Model

# DANN:
# - Stress classification
# - Domain classification

class DomainGeneralisationModel(nn.Module):

    def __init__(self, num_domains=3):

        super().__init__()

        # Shared backbone
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        self.feature_extractor = backbone.features

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Stress classifier
        self.stress_classifier = nn.Sequential(

            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # Domain classifier
        self.gradient_reversal = GradientReversalLayer(
            lambda_=0.0
        )

        self.domain_classifier = nn.Sequential(

            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_domains)
        )

    def forward(self, x):

        # Feature extraction
        features = self.feature_extractor(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)

        # Stress prediction
        stress_out = self.stress_classifier(features)

        # Domain prediction
        reversed_features = self.gradient_reversal(features)

        domain_out = self.domain_classifier(
            reversed_features
        )

        return stress_out, domain_out

#-----------------------------------------------------------------------------------------------------------------------

# Section 8 - Gradient Reversal Layer

# Used in Domain Generalisation (DANN)

class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):

        ctx.save_for_backward(
            torch.tensor(
                lambda_,
                dtype=torch.float32
            )
        )

        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):

        lambda_ = ctx.saved_tensors[0].item()

        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):

    def __init__(self, lambda_=0.0):

        super().__init__()

        self.lambda_ = lambda_

    def forward(self, x):

        return GradientReversalFunction.apply(
            x,
            self.lambda_
        )

#-----------------------------------------------------------------------------------------------------------------------