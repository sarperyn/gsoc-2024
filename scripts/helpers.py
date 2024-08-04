import os
import sys
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import cv2  # opencv+python == 4.9.0.80 # image processing
import matplotlib.pyplot as plt
import hdf5storage  # 0.1.19 # for loading .mat files

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import seaborn as sns
from collections import Counter


def load_and_preprocess_data(data_dir, image_dimension=256):
    images = []
    masks = []
    labels = []
    planes = []
    files = [f for f in os.listdir(data_dir) if f != '.DS_Store'] # ignore any .DS_Store files on MacOS

    # Data integrity check
    print(f"Total files found: {len(files)}")
    processed_count = 0

    for i, file in enumerate(files, start=1):
        try:
            mat_file = hdf5storage.loadmat(os.path.join(data_dir, file))['cjdata'][0] # Load the .mat file

            # Resize and normalize the images
            image = mat_file['image']
            image = cv2.resize(image, dsize=(image_dimension, image_dimension), interpolation=cv2.INTER_CUBIC)
            image = image.astype(np.float32) / 255.0  # Scale image to range [0, 1]
            image = np.expand_dims(image, axis=-1)

            # Resize and prepare mask for multi-class segmentation
            mask = mat_file['tumorMask'].astype('uint8')
            mask = cv2.resize(mask, dsize=(image_dimension, image_dimension), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1) # add the channel dimension (prep for check later)

            # Get the label and convert to one-hot
            label = int(mat_file['label'])
            plane = identify_plane(file)  # identify the plane from the file name
            
            one_hot_mask = np.zeros((image_dimension, image_dimension, 3), dtype=np.float32)  # three classes # set dtype to float32
            for j in range(1, 4):  # labels are 1, 2, 3
                one_hot_mask[:, :, j-1] = (mask[:, :, 0] == j).astype(np.float32)  # Cast to float32

            images.append(image)
            masks.append(one_hot_mask)
            labels.append(label)
            planes.append(plane)
            processed_count += 1

            if i % 10 == 0:
                sys.stdout.write(f'\r[{i}/{len(files)}] images loaded: {i / float(len(files)) * 100:.1f} %')
                sys.stdout.flush()

        except Exception as e:
            print(f"Failed to process file {file}: {e}")

    print(f"\nFinished loading and processing data. Successfully processed {processed_count}/{len(files)} files.")
    return np.array(images), np.array(masks), np.array(labels), np.array(planes)



def identify_plane(file_name):
    if 'axial' in file_name.lower():
        return 'axial'
    elif 'coronal' in file_name.lower():
        return 'coronal'
    elif 'sagittal' in file_name.lower():
        return 'sagittal'
    else:
        return file_name  # debug check: return the file name if the plane type is unknown



def visualize_sample_images(images, masks, labels, planes):
    label_to_tumor = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary Tumor'}
    plane_order = ['axial', 'coronal', 'sagittal']
    tumor_order = [1, 2, 3]

    samples = {plane: {label: None for label in tumor_order} for plane in plane_order}

    for idx, label in enumerate(labels):
        plane = planes[idx]
        if samples[plane][label] is None:
            samples[plane][label] = (images[idx], masks[idx], label)
        if all(all(samples[plane][label] is not None for label in tumor_order) for plane in plane_order):
            break

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row, plane in enumerate(plane_order):
        for col, label in enumerate(tumor_order):
            if samples[plane][label]:
                image, mask, label = samples[plane][label]
                tumor_type = label_to_tumor[label]
                title = f"{plane.capitalize()} - {tumor_type}"

                combined_mask = np.zeros_like(mask[..., 0])
                for i in range(mask.shape[-1]):
                    combined_mask += (mask[..., i] * (i + 1)).astype(np.uint8)

                ax = axes[row, col]
                ax.imshow(image.squeeze(), cmap='gray')
                ax.imshow(combined_mask, cmap='viridis', vmin=0, vmax=3, alpha=0.5)
                ax.set_title(title)
                ax.axis('off')

    plt.tight_layout()
    plt.show()

# debug check (recommended)
def check_data_distribution(labels):
    label_to_tumor = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary Tumor'}
    label_counts = Counter(labels)
    total_counts = len(labels)
    for label, count in label_counts.items():
        tumor_type = label_to_tumor.get(label, "Unknown Tumor Type")
        print(f"Label {label} ({tumor_type}): {count} slices, {count / total_counts * 100:.2f}% of the dataset")



############
def filter_data(images, masks, labels, planes, file_names, tumor_type='all', plane_type='all'):
    """Filter the dataset based on the selected tumor type and plane type."""
    
    # Check for valid tumor type and plane type
    if tumor_type not in ['all', 'meningioma', 'glioma', 'pituitary']:
        raise ValueError("Invalid tumor type. Choose from 'all', 'meningioma', 'glioma', 'pituitary'.")
    
    if plane_type not in ['all', 'axial', 'coronal', 'sagittal']:
        raise ValueError("Invalid plane type. Choose from 'all', 'axial', 'coronal', 'sagittal'.")
    
    # Filter by tumor type
    if tumor_type != 'all':
        tumor_type_to_label = {'meningioma': 1, 'glioma': 2, 'pituitary': 3}
        selected_label = tumor_type_to_label[tumor_type]
        tumor_filtered_indices = [i for i, lbl in enumerate(labels) if lbl == selected_label]
    else:
        tumor_filtered_indices = range(len(labels))
    
    # Filter by plane type
    if plane_type != 'all':
        plane_filtered_indices = [i for i in tumor_filtered_indices if planes[i] == plane_type]
    else:
        plane_filtered_indices = tumor_filtered_indices
    
    images = [images[i] for i in plane_filtered_indices]
    masks = [masks[i] for i in plane_filtered_indices]
    labels = [labels[i] for i in plane_filtered_indices]
    planes = [planes[i] for i in plane_filtered_indices]
    file_names = [file_names[i] for i in plane_filtered_indices]

    return images, masks, labels, planes, file_names

def split_data(images, masks, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into training, validation, and testing sets.
    - test_size specifies the proportion of the data for testing.
    - val_size specifies the proportion of the training data for validation.
    """
    
    # entire dataset is split into training and testing sets
    images_train, images_test, masks_train, masks_test, labels_train, labels_test = train_test_split(
        images, masks, labels, test_size=test_size, random_state=random_state, stratify=labels)

    # training set obtained from the first split is further split into training and validation sets
    images_train, images_val, masks_train, masks_val, labels_train, labels_val = train_test_split(
        images_train, masks_train, labels_train, test_size=val_size, random_state=random_state, stratify=labels_train)

    return (np.array(images_train), np.array(masks_train), np.array(labels_train),
            np.array(images_val), np.array(masks_val), np.array(labels_val),
            np.array(images_test), np.array(masks_test), np.array(labels_test))


def mean_dice_coef(true_mask, pred_mask, smooth=1.0, num_classes=3):
    dice_scores = []
    for class_idx in range(num_classes):
        class_true_mask = true_mask[:, class_idx, ...]
        class_pred_mask = pred_mask[:, class_idx, ...]
        intersection = torch.sum(class_true_mask * class_pred_mask)
        sum_true_mask = torch.sum(class_true_mask)
        sum_pred_mask = torch.sum(class_pred_mask)
        dice_score = (2. * intersection + smooth) / (sum_true_mask + sum_pred_mask + smooth)
        dice_scores.append(dice_score)
    mean_dice = torch.mean(torch.tensor(dice_scores))
    return mean_dice

def dice_coef_class_0(true_mask, pred_mask, smooth=1.0):
    class_true_mask = true_mask[:, 0, ...]
    class_pred_mask = pred_mask[:, 0, ...]
    intersection = torch.sum(class_true_mask * class_pred_mask)
    sum_true_mask = torch.sum(class_true_mask)
    sum_pred_mask = torch.sum(class_pred_mask)
    dice_score = (2. * intersection + smooth) / (sum_true_mask + sum_pred_mask + smooth)
    return dice_score

def dice_coef_class_1(true_mask, pred_mask, smooth=1.0):
    class_true_mask = true_mask[:, 1, ...]
    class_pred_mask = pred_mask[:, 1, ...]
    intersection = torch.sum(class_true_mask * class_pred_mask)
    sum_true_mask = torch.sum(class_true_mask)
    sum_pred_mask = torch.sum(class_pred_mask)
    dice_score = (2. * intersection + smooth) / (sum_true_mask + sum_pred_mask + smooth)
    return dice_score

def dice_coef_class_2(true_mask, pred_mask, smooth=1.0):
    class_true_mask = true_mask[:, 2, ...]
    class_pred_mask = pred_mask[:, 2, ...]
    intersection = torch.sum(class_true_mask * class_pred_mask)
    sum_true_mask = torch.sum(class_true_mask)
    sum_pred_mask = torch.sum(class_pred_mask)
    dice_score = (2. * intersection + smooth) / (sum_true_mask + sum_pred_mask + smooth)
    return dice_score


def mean_iou_coef(true_mask, pred_mask, smooth=1.0, num_classes=3):
    iou_scores = []
    for class_idx in range(num_classes):
        class_true_mask = true_mask[:, class_idx, ...]
        class_pred_mask = pred_mask[:, class_idx, ...]
        intersection = torch.sum(class_true_mask * class_pred_mask)
        sum_true_mask = torch.sum(class_true_mask)
        sum_pred_mask = torch.sum(class_pred_mask)
        union = sum_true_mask + sum_pred_mask - intersection
        iou_score = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou_score)
    mean_iou = torch.mean(torch.tensor(iou_scores))
    return mean_iou

def iou_coef_class_0(true_mask, pred_mask, smooth=1.0):
    class_true_mask = true_mask[:, 0, ...]
    class_pred_mask = pred_mask[:, 0, ...]
    intersection = torch.sum(class_true_mask * class_pred_mask)
    sum_true_mask = torch.sum(class_true_mask)
    sum_pred_mask = torch.sum(class_pred_mask)
    union = sum_true_mask + sum_pred_mask - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score

def iou_coef_class_1(true_mask, pred_mask, smooth=1.0):
    class_true_mask = true_mask[:, 1, ...]
    class_pred_mask = pred_mask[:, 1, ...]
    intersection = torch.sum(class_true_mask * class_pred_mask)
    sum_true_mask = torch.sum(class_true_mask)
    sum_pred_mask = torch.sum(class_pred_mask)
    union = sum_true_mask + sum_pred_mask - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score

def iou_coef_class_2(true_mask, pred_mask, smooth=1.0):
    class_true_mask = true_mask[:, 2, ...]
    class_pred_mask = pred_mask[:, 2, ...]
    intersection = torch.sum(class_true_mask * class_pred_mask)
    sum_true_mask = torch.sum(class_true_mask)
    sum_pred_mask = torch.sum(class_pred_mask)
    union = sum_true_mask + sum_pred_mask - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score




# Dice Loss
def dice_loss(true_mask, pred_mask, smooth=1.0):
    intersection = torch.sum(true_mask * pred_mask)
    sum_true_mask = torch.sum(true_mask)
    sum_pred_mask = torch.sum(pred_mask)
    dice_loss_value = 1 - (2. * intersection + smooth) / (sum_true_mask + sum_pred_mask + smooth)
    return dice_loss_value

# Combined Loss for a single class
def combined_loss_class(true_mask, pred_mask, class_idx, weight_ce=0.1, weight_dice=0.9):
    true_class = true_mask[:, class_idx, ...]
    pred_class = pred_mask[:, class_idx, ...]
    ce_loss = F.cross_entropy(pred_class, true_class.long(), reduction='none')
    dice_loss_value = dice_loss(true_class, pred_class)
    combined_loss_value = weight_ce * ce_loss + weight_dice * dice_loss_value
    class_loss = torch.mean(combined_loss_value)
    return class_loss

# # Combined Loss for class 0
# def combined_loss_class_0(true_mask, pred_mask, weight_ce=0.1, weight_dice=0.9):
#     return combined_loss_class(true_mask, pred_mask, 0, weight_ce, weight_dice)

# # Combined Loss for class 1
# def combined_loss_class_1(true_mask, pred_mask, weight_ce=0.1, weight_dice=0.9):
#     return combined_loss_class(true_mask, pred_mask, 1, weight_ce, weight_dice)

# # Combined Loss for class 2
# def combined_loss_class_2(true_mask, pred_mask, weight_ce=0.1, weight_dice=0.9):
    return combined_loss_class(true_mask, pred_mask, 2, weight_ce, weight_dice)

# Mean Combined Loss across all classes
def mean_combined_loss(true_mask, pred_mask, weight_ce=0.1, weight_dice=0.9, num_classes=3):
    class_losses = []
    for i in range(num_classes):
        class_loss = combined_loss_class(true_mask, pred_mask, i, weight_ce, weight_dice)
        class_losses.append(class_loss)
    mean_loss = torch.mean(torch.stack(class_losses), dim=0)
    return mean_loss


def mean_accuracy_metric(true_mask, pred_mask, num_classes=3):
    accuracies = []
    for i in range(num_classes):
        true_class = (torch.argmax(true_mask, dim=1) == i)
        pred_class = (torch.argmax(pred_mask, dim=1) == i)
        correct_predictions = torch.sum((true_class & pred_class).float())
        total_predictions = torch.sum(true_class.float())
        accuracy = correct_predictions / (total_predictions + 1e-7)
        accuracies.append(accuracy)
    mean_accuracy = torch.mean(torch.tensor(accuracies))
    return mean_accuracy

def accuracy_class_0(true_mask, pred_mask):
    true_class = (torch.argmax(true_mask, dim=1) == 0)
    pred_class = (torch.argmax(pred_mask, dim=1) == 0)
    correct_predictions = torch.sum((true_class & pred_class).float())
    total_predictions = torch.sum(true_class.float())
    accuracy = correct_predictions / (total_predictions + 1e-7)
    return accuracy

def accuracy_class_1(true_mask, pred_mask):
    true_class = (torch.argmax(true_mask, dim=1) == 1)
    pred_class = (torch.argmax(pred_mask, dim=1) == 1)
    correct_predictions = torch.sum((true_class & pred_class).float())
    total_predictions = torch.sum(true_class.float())
    accuracy = correct_predictions / (total_predictions + 1e-7)
    return accuracy

def accuracy_class_2(true_mask, pred_mask):
    true_class = (torch.argmax(true_mask, dim=1) == 2)
    pred_class = (torch.argmax(pred_mask, dim=1) == 2)
    correct_predictions = torch.sum((true_class & pred_class).float())
    total_predictions = torch.sum(true_class.float())
    accuracy = correct_predictions / (total_predictions + 1e-7)
    return accuracy


class MetricsCollector:
    def __init__(self, model, validation_data, num_classes=3, device='cuda'):
        self.model = model
        self.validation_data = validation_data
        self.num_classes = num_classes
        self.metrics_history = []
        self.device = device

    def on_epoch_end(self, epoch):
        val_images, true_mask = self.validation_data
        val_images, true_mask = val_images.to(self.device), true_mask.to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_mask = self.model(val_images)
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = F.one_hot(pred_mask, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if true_mask.shape[1] == 1:
            true_mask = torch.squeeze(true_mask, dim=1)
            true_mask = F.one_hot(true_mask.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        elif true_mask.shape[1] != self.num_classes:
            true_mask = F.one_hot(true_mask.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        mean_acc, class_accs = self.calculate_accuracies(true_mask, pred_mask)
        mean_loss = mean_combined_loss(true_mask, pred_mask, self.num_classes)
        dice_scores, mean_dice = self.calculate_dice_coef(true_mask, pred_mask)
        iou_scores, mean_iou = self.calculate_iou_coef(true_mask, pred_mask)

        logs = {
            'val_mean_accuracy_metric': mean_acc.item(),
            'val_mean_combined_loss': mean_loss.item(),
            'val_mean_dice_coef': mean_dice.item(),
            'val_mean_iou_coef': mean_iou.item()
        }

        for i in range(self.num_classes):
            logs[f'val_accuracy_class_{i}'] = class_accs[i].item()
            logs[f'val_dice_coef_class_{i}'] = dice_scores[i].item()
            logs[f'val_iou_coef_class_{i}'] = iou_scores[i].item()
            class_loss = self.calculate_combined_loss(true_mask, pred_mask, i)
            logs[f'val_combined_loss_class_{i}'] = class_loss.item()

        self.metrics_history.append(logs)
        print(f"Epoch {epoch + 1} validation metrics collected.")

    def calculate_accuracies(self, true_mask, pred_mask):
        class_accs = []
        for i in range(self.num_classes):
            class_true_mask = true_mask[:, i, ...]
            class_pred_mask = pred_mask[:, i, ...]
            correct_predictions = torch.sum(class_true_mask * class_pred_mask)
            total_predictions = torch.sum(class_true_mask)
            class_acc = correct_predictions / (total_predictions + 1e-7)
            class_accs.append(class_acc)
        mean_acc = torch.mean(torch.tensor(class_accs))
        return mean_acc, class_accs

    def calculate_dice_coef(self, true_mask, pred_mask, smooth=1.0):
        dice_scores = []
        for class_idx in range(self.num_classes):
            class_true_mask = true_mask[:, class_idx, ...]
            class_pred_mask = pred_mask[:, class_idx, ...]
            intersection = torch.sum(class_true_mask * class_pred_mask)
            sum_true_mask = torch.sum(class_true_mask)
            sum_pred_mask = torch.sum(class_pred_mask)
            dice_score = (2. * intersection + smooth) / (sum_true_mask + sum_pred_mask + smooth)
            dice_scores.append(dice_score)
        mean_dice = torch.mean(torch.tensor(dice_scores))
        return dice_scores, mean_dice

    def calculate_iou_coef(self, true_mask, pred_mask, smooth=1.0):
        iou_scores = []
        for class_idx in range(self.num_classes):
            class_true_mask = true_mask[:, class_idx, ...]
            class_pred_mask = pred_mask[:, class_idx, ...]
            intersection = torch.sum(class_true_mask * class_pred_mask)
            sum_true_mask = torch.sum(class_true_mask)
            sum_pred_mask = torch.sum(class_pred_mask)
            union = sum_true_mask + sum_pred_mask - intersection
            iou_score = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou_score)
        mean_iou = torch.mean(torch.tensor(iou_scores))
        return iou_scores, mean_iou

    def calculate_combined_loss(self, true_mask, pred_mask, class_idx, weight_ce=0.1, weight_dice=0.9):
        true_class = true_mask[:, class_idx, ...]
        pred_class = pred_mask[:, class_idx, ...]
        ce_loss = F.cross_entropy(pred_class.unsqueeze(1), true_class.long(), reduction='none')
        dice_loss_value = dice_loss(true_class, pred_class)
        combined_loss_value = weight_ce * ce_loss + weight_dice * dice_loss_value
        class_loss = torch.mean(combined_loss_value)
        return class_loss

    def on_train_end(self):
        metrics_df = pd.DataFrame(self.metrics_history)
        print("Training Metrics per Epoch:")
        print(metrics_df.to_string(index=False))
        self.metrics_df = metrics_df

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random



# Function to get augmentation parameters
def get_augmentation_transforms(is_augmented):
    if is_augmented:
        return transforms.Compose([
            transforms.RandomRotation(45),  # rotate left-right by x degrees (0-360 degrees max, 0-90 typical)
            transforms.RandomResizedCrop(256, scale=(0.75, 1.25)),  # zoom
            transforms.RandomHorizontalFlip(),  # yes/no flipped horizontally
            transforms.RandomAffine(degrees=0, shear=5),  # shear
            transforms.ColorJitter(),  # color jitter
            transforms.ToTensor(),
        ])
    else:
        return transforms.ToTensor()

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)  # apply this seed to transforms
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        
        return image, mask

# Function to create data loader
def create_data_loader(images, masks, batch_size, transform=None):
    dataset = SegmentationDataset(images, masks, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Function to visualize original and augmented images
def compare_original_and_augmented(original_loader, augmented_loader, num_samples=1):
    augmentation_params = get_augmentation_transforms(True)  # True because we need the full set for description
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    transformations = [
        "Rotation: ±45°",
        "Width Shift: ±25%",
        "Height Shift: ±25%",
        "Shear: ±5°",
        "Zoom: 75-125%",
        "Horizontal Flip",
        "Color Jitter",
        "Fill mode: 'nearest'"
    ]
    transformation_description = "Applied Transformations:\n" + ", ".join(transformations)
    fig.suptitle(transformation_description, fontsize=15)

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        orig_images_batch, orig_masks_batch = next(iter(original_loader))
        aug_images_batch, aug_masks_batch = next(iter(augmented_loader))

        axes[i, 0].imshow(orig_images_batch[0].permute(1, 2, 0), cmap='gray')  # img 1 - original img
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(orig_masks_batch[0].permute(1, 2, 0), cmap='gray')  # img 2 - original mask
        axes[i, 1].set_title('Original Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(aug_images_batch[0].permute(1, 2, 0), cmap='gray')  # img 3 - augmented img
        axes[i, 2].set_title('Augmented Image')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(aug_masks_batch[0].permute(1, 2, 0), cmap='gray')  # img 4 - augmented mask
        axes[i, 3].set_title('Augmented Mask')
        axes[i, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# Function to check the shape of augmented data
def check_augmented_data_shape(images_train, masks_train, batch_size, is_augmented):
    if is_augmented:
        print("Checking first batch output:")
        loader = create_data_loader(images_train, masks_train, batch_size, transform=get_augmentation_transforms(True))
        sample_img_output, sample_mask_output = next(iter(loader))
        
        assert sample_img_output.shape == (batch_size, 1, 256, 256), \
            f"Augmented image shape mismatch. Expected: (batch_size, channels, height, width), Got: {sample_img_output.shape}"
        assert sample_mask_output.shape == (batch_size, 1, 256, 256), \
            f"Augmented mask shape mismatch. Expected: (batch_size, channels, height, width), Got: {sample_mask_output.shape}"
        print(f"Augmented image batch shape is correct: {sample_img_output.shape}")
        print(f"Augmented mask batch shape is correct: {sample_mask_output.shape}")
    else:
        print(f"Images shape: {images_train.shape}")
        print(f"Masks shape: {masks_train.shape}")
        assert images_train.shape[0] == masks_train.shape[0], "Mismatch in number of images and masks."
        print("Non-augmented data shapes are correct.")

# Convert images and masks to PyTorch tensors


### ACTION REQUIRED ###

path = '/home/syurtseven/GI-Tract-Image-Segmentation/synthetic_data_generation/gsoc/data' # local runs
data_dir = os.path.join(path, 'brain') # local runs

files = os.listdir(data_dir)
print(f"Files in {data_dir}:", len(files))
image_dimension = 256

tumor_type = 'glioma'  # change to 'all', 'meningioma', 'glioma', or 'pituitary'
plane_type = 'sagittal'  # change to 'all', 'axial', 'coronal', or 'sagittal'

images, masks, labels, planes = load_and_preprocess_data(data_dir, image_dimension)

file_names = [f for f in os.listdir(data_dir) if f != '.DS_Store'] # ignore .DS store on Macs

# Filter the data
images, masks, labels, planes, file_names = filter_data(images, masks, labels, planes, file_names, tumor_type, plane_type)

# Debug check: does plane selection match file name?
label_to_tumor = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary Tumor'}
for file_name, label, plane in zip(file_names, labels, planes):
    print(f"File: {file_name}, Label: {label_to_tumor[label]}, Plane: {plane}")


check_data_distribution(labels)

# Data Augmentation (optional, recommended)
is_augmented = True  # activate augmentation Y/N?

compare_original_and_augmented(original_loader, augmented_loader, num_samples=1)

transform = get_augmentation_transforms(is_augmented)

original_loader = create_data_loader(images_train, masks_train, batch_size, transform=None)
augmented_loader = create_data_loader(images_train, masks_train, batch_size, transform=transform)


images_train = torch.tensor(images_train).unsqueeze(1).float()
masks_train = torch.tensor(masks_train).unsqueeze(1).float()

check_augmented_data_shape(images_train, masks_train, batch_size, is_augmented)