#-----------------------------------------------------------------------------------------------------------------------

# Section 1 - Import Libraries

import os
import cv2
import torch
import numpy as np
from torch.utils.data import (Dataset, DataLoader, ConcatDataset)
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------------------------------------------------

#Section 2 - Custom Dataset Class

#This willl load images from folders.

class LeafDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        else:
            image = image.astype("uint8")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
#-----------------------------------------------------------------------------------------------------------------------

#Section 3 - Load Dataset Paths and Labels

#This section collects all image paths and labels and label mapping will be 0 - Non Stress and 1 - Streess.
# 1. Tomato Dataset
# 2. Maize Folder Dataset
# 3. Maize .npy Dataset

# Tomato Dataset Mapping

# 100%_field_capacity → Non-stress (0)
# 75%_field_capacity  → Stress (1)
# 50%_field_capacity  → Stress (1)
# 25%_field_capacity  → Stress (1)

def load_tomato_dataset(dataset_path):
    image_paths = []
    labels = []

    stress_folders = [
        "25%_field_capacity",
        "50%_field_capacity",
        "75%_field_capacity"
    ]

    non_stress_folders = [
        "100%_field_capacity"
    ]

    for folder in os.listdir(dataset_path):
        if folder.startswith("."):
            continue
        folder_path = os.path.join(dataset_path, folder)

        # Skip if not directory
        if not os.path.isdir(folder_path):
            continue

        #Assign Labels
        if folder in stress_folders:
            label = 1  # Stress
        else:
            label = 0  # Non-stress

        for img_file in os.listdir(folder_path):
            if img_file.startswith("."):
                continue

            img_path = os.path.join(folder_path, img_file)
            image_paths.append(img_path)
            labels.append(label)
            
    return image_paths, labels


# Maize Folder Dataset Mapping

# WW  → Non-stress (0)
# MIS → Stress (1)
# MOD → Stress (1)
# WS  → Stress (1)

def load_maize_dataset(dataset_path):

    image_paths = []
    labels = []

    stress_folders = ["MIS", "MOD", "WS"]
    non_stress_folders = ["WW"]

    for folder in os.listdir(dataset_path):

        if folder.startswith("."):
            continue

        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        # Assign labels
        if folder in stress_folders:
            label = 1
        else:
            label = 0

        for img_file in os.listdir(folder_path):

            if img_file.startswith("."):
                continue

            img_path = os.path.join(folder_path, img_file)

            image_paths.append(img_path)
            labels.append(label)

    return image_paths, labels


#Load .npy maize dataset

#In this the important task is to check the format of lables for example 0 and 1 or labels are multicalss if that they need to be converted to binary labels.

npy_dataset_path = "/Users/manish/Documents/Semister 2/7. Project/Dataset/MaizeLeaf_dataset"

# Load arrays
train_maize2_data = np.load(os.path.join(npy_dataset_path, "trainData.npy"))
train_maize2_labels = np.load(os.path.join(npy_dataset_path, "trainLabel.npy"))

val_maize2_data = np.load(os.path.join(npy_dataset_path, "valData.npy"))
val_maize2_labels = np.load(os.path.join(npy_dataset_path, "valLabel.npy"))

test_maize2_data = np.load(os.path.join(npy_dataset_path, "testData.npy"))
test_maize2_labels = np.load(os.path.join(npy_dataset_path, "testLabel.npy"))

print("Train Shape:", train_maize2_data.shape)
print("Validation Shape:", val_maize2_data.shape)
print("Test Shape:", test_maize2_data.shape)

print("Unique Labels:", np.unique(train_maize2_labels))


# Original Labels:
# 0 - Well-watered
# 1 - Reduced-watered
# 2 - Drought-stressed


train_maize2_labels = np.where(train_maize2_labels == 0, 0, 1)

val_maize2_labels = np.where(val_maize2_labels == 0, 0, 1)

test_maize2_labels = np.where(test_maize2_labels == 0, 0, 1)

# Check converted labels
print("Converted Train Labels:", np.unique(train_maize2_labels))
print("Converted Validation Labels:", np.unique(val_maize2_labels))
print("Converted Test Labels:", np.unique(test_maize2_labels))


# Check class distribution

print("Train Label Distribution:")
print(np.bincount(train_maize2_labels))

print("Validation Label Distribution:")
print(np.bincount(val_maize2_labels))

print("Test Label Distribution:")
print(np.bincount(test_maize2_labels))

#-----------------------------------------------------------------------------------------------------------------------

#Section 4 - Load all datasets

tomato_image_path = "/Users/manish/Documents/Semister 2/7. Project/Dataset/Tomato Plant dataset/dataset"
maize_image_path = "/Users/manish/Documents/Semister 2/7. Project/Dataset/Maize Water Stress"

# Load Tomato Dataset
tomato_images, tomato_labels = load_tomato_dataset(tomato_image_path)

# Load Maize Dataset
maize_images, maize_labels = load_maize_dataset(maize_image_path)

print("Tomato Images:", len(tomato_images))
print("Maize Images:", len(maize_images))

print("Tomato Labels:", np.unique(tomato_labels))
print("Maize Labels:", np.unique(maize_labels))

#-----------------------------------------------------------------------------------------------------------------------

#Section 5 - Train Test Split

#Tomato Dataset Split

train_tomato_data, temp_tomato_data, train_tomato_labels, temp_tomato_labels = train_test_split(
    tomato_images,
    tomato_labels,
    test_size=0.30,
    random_state=42,
    stratify=tomato_labels
)

val_tomato_data, test_tomato_data, val_tomato_labels, test_tomato_labels = train_test_split(
    temp_tomato_data,
    temp_tomato_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_tomato_labels
)

print("Tomato Train:", len(train_tomato_data))
print("Tomato Validation:", len(val_tomato_data))
print("Tomato Test:", len(test_tomato_data))

#Maize Dataset Split

train_maize_data, temp_maize_data, train_maize_labels, temp_maize_labels = train_test_split(
    maize_images,
    maize_labels,
    test_size=0.30,
    random_state=42,
    stratify=maize_labels
)

val_maize_data, test_maize_data, val_maize_labels, test_maize_labels = train_test_split(
    temp_maize_data,
    temp_maize_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_maize_labels
)

print("Maize Train:", len(train_maize_data))
print("Maize Validation:", len(val_maize_data))
print("Maize Test:", len(test_maize_data))

#-----------------------------------------------------------------------------------------------------------------------

# Section 6 - Domain Labelled Datasets

#Added the domian_id so which tells the model which dataset the image is coming from.

class DomainLeafDataset(Dataset):
    """
    Extends LeafDataset with a domain_id per sample.
    domain_id: 0 = Tomato, 1 = Maize, 2 = Maize2
    
    Why domain_id matters: The DANN models domain classifier needs this label
    to learn which domain each image belongs to — and then the gradient reversal
    forces the feature extractor to unlearn those domain cues.
    """
 
    def __init__(self, images, labels, domain_id, transform=None):
        self.images = images
        self.labels = labels
        self.domain_id = domain_id
        self.transform = transform
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image.astype("uint8")
 
        if self.transform:
            image = self.transform(image)
 
        # Return domain_id as a tensor scalar alongside image and stress label
        return image, label, torch.tensor(self.domain_id, dtype=torch.long)
    
#-----------------------------------------------------------------------------------------------------------------------
