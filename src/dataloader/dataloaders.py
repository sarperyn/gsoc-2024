import torch
import cv2
import os
import glob
from typing import Any
# import sys
# sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2

from ..utils.variable_utils import MADISON_DATA
from ..utils.data_utils import read_image, resize_torch_tensor


class MadisonDataset(Dataset):

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
        

if __name__ == '__main__':

    image_paths = glob.glob(os.path.join(MADISON_DATA, '*'))
    print(f"#IMAGES:{len(image_paths)}")

    dataset    = MadisonDataset(image_paths=image_paths)
    dataloader = DataLoader(dataset=dataset, batch_size=30)
    print(f"#BATCHES:{len(dataloader)}")
    data = next(iter(dataloader))
    print(data.shape)

