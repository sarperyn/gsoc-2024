import torch
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from PIL import Image
# import sys
# sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2

from ..utils.variable_utils import MADISON_DATA
from ..utils.data_utils import read_image, resize_torch_tensor


class DatasetNoLabels(Dataset):

    def __init__(self, image_paths) -> None:
        
        self.image_paths = image_paths
        self.filtered_paths = []

        img_dict = {}
        for img in self.image_paths:

            re = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if len(re.shape) == 2 :
                self.filtered_paths.append(img)

    def __len__(self) -> int:
        return len(self.filtered_paths)

    def __getitem__(self, index) -> Any:

        #img = cv2.imread(self.image_paths[index], cv2.IMREAD_UNCHANGED)
        img = read_image(self.filtered_paths[index])
        img = resize_torch_tensor(img, 256, 256)

        return img
        
class DatasetLabeled(Dataset):

    def __init__(self, segmentation_path, args, augment=False) -> None:
        self.image_paths = sorted(glob.glob(os.path.join(segmentation_path, '*image*.png')))
        self.mask_paths  = sorted(glob.glob(os.path.join(segmentation_path, '*mask*.png')))
        self.args        = args

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

        self.augment = augment
        if self.augment:
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])

    def get_cases(self, n_cases=10):

        case_set = set()
        for img in self.image_paths:
            case_num = img.split('/')[-1].split('_')[0]
            case_set.add(case_num)

        val_cases = list(case_set)[:n_cases]
        return val_cases

    def split_data(self, val_cases):

        training   = []
        validation = []

        for i in range(len(self.image_paths)):

            case_num = self.image_paths[i].split('/')[-1].split('_')[0]

            if case_num in val_cases:
                validation.append(self.image_paths[i])
                validation.append(self.mask_paths[i])
            
            else:
                training.append(self.image_paths[i])
                training.append(self.mask_paths[i])

        return training, validation
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)
        
        # Apply the transformations
        img = self.transform(img)
        mask = self.mask_transform(mask)

        if self.augment:
            seed = torch.seed() 
            torch.manual_seed(seed)
            img = self.augmentation_transforms(img)
            torch.manual_seed(seed)
            mask = self.augmentation_transforms(mask)

        return img, mask, self.image_paths[index]

if __name__ == '__main__':

    # image_paths = glob.glob(os.path.join(MADISON_DATA, '*'))
    # print(f"#IMAGES:{len(image_paths)}")

    # dataset    = MadisonDataset(image_paths=image_paths)
    # dataloader = DataLoader(dataset=dataset, batch_size=30)
    # print(f"#BATCHES:{len(dataloader)}")
    # data = next(iter(dataloader))
    # print(data.shape)

    dataset    = DatasetLabeled(segmentation_path='/home/syurtseven/gsoc-2024/data/stomach_masks',augment=False)
    dataloader = DataLoader(dataset=dataset, batch_size=5)
    data = next(iter(dataloader))

    img, mask = data
    print(len(dataloader))
    print(img.shape)
    print(mask.shape)