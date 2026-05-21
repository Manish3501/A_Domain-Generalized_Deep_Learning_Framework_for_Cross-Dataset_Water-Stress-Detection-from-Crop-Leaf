#Section 3 -  Ensemble Image Transformations 

#This section is for preprocessing all images before training so it resize all images to same size, normalizes the pixel values to be between 0
#and convert images from pixels.

from torchvision import transforms


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#Section 1 - Ensemble Improved Data Augmentation

#Added some augmention to improve the generalization of model and to prevent overfitting.

improved_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.RandomResizedCrop(
        224,
        scale=(0.9,1.0)
    ),

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



# Section 1 - (Domain Generalization)Domain Augmentation

dg_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),          # Extra: leaves can appear upside down
    transforms.RandomRotation(20),                  # Slightly wider than your 15° — more robust
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(
        brightness=0.3,                             # Wider range than your 0.2 — handles
        contrast=0.3,                               # different camera/lighting per dataset
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
 
# Validation/test transform — no augmentation, only resize + normalise
# (same as your original transform, no random ops so results are reproducible)
dg_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])