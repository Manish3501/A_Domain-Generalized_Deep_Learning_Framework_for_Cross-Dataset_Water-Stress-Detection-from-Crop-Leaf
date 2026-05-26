#-----------------------------------------------------------------------------------------------------------------------

# Section 1 - Import Libraries

from torchvision import transforms

#-----------------------------------------------------------------------------------------------------------------------

#Section 2 -  Ensemble Image Transformations 

#This section is for preprocessing all images before training so it resize all images to same size, 
# normalizes the pixel values to be between 0 and 1, and converts images from pixels to tensors.

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#-----------------------------------------------------------------------------------------------------------------------

#Section 3 - Improved Ensemble Transformations

#Added some augmention to improve the generalization of model and to prevent overfitting.

improved_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.ColorJitter( 
        brightness=0.2, 
        contrast=0.2
        ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#-----------------------------------------------------------------------------------------------------------------------

# Section 4 - Domain Generalization Transformations

dg_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#-----------------------------------------------------------------------------------------------------------------------

# Section 5 -  Validation/test transform

# No augmentation, only resize + normalise

dg_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#-----------------------------------------------------------------------------------------------------------------------

#Reference for normalization values - https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

#-----------------------------------------------------------------------------------------------------------------------