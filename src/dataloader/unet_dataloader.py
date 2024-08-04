
import torch
import cv2
import os
import glob
import numpy as np
import hdf5storage 
import sys
from collections import Counter
from typing import Any
# import sys
# sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..utils.variable_utils import BRAIN_DATA
from ..utils.data_utils import min_max_normalize


class BrainDataset(Dataset):

    def load_and_preprocess_data(self, data_dir, image_dimension=256):
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
                plane = self.identify_plane(file)  # identify the plane from the file name
                
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


        return np.stack(images, axis=0), np.stack(masks, axis=0), np.stack(labels, axis=0), planes

    def identify_plane(self, file_name):
        if 'axial' in file_name.lower():
            return 'axial'
        elif 'coronal' in file_name.lower():
            return 'coronal'
        elif 'sagittal' in file_name.lower():
            return 'sagittal'
        else:
            return file_name  # debug check: return the file name if the plane type is unknown


    def filter_data(self, images, masks, labels, planes, file_names, tumor_type='all', plane_type='all'):
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
    

    def check_dimensions(self, images, masks, required_image_shape=(1, 256, 256), required_mask_shape=(3, 256, 256)):
        assert images.shape == required_image_shape, f"Image shape mismatch: expected {required_image_shape}, got {images.shape}"
        assert masks.shape == required_mask_shape, f"Mask shape mismatch: expected {required_mask_shape}, got {masks.shape}"
        print(f"Image shape is {images.shape} and mask shape is {masks.shape}.")
        print("All images and masks correctly match the required shapes.")

    def __init__(self, data_dir, img_dim=256) -> None:

        self.data_dir = data_dir
        self.images_path = glob.glob(os.path.join(self.data_dir,'*'))
        self.images, self.masks, self.labels, self.planes = self.load_and_preprocess_data(data_dir, 
                                                                      img_dim)
        
        self.images = self.images.transpose((0, 3, 1, 2))
        self.masks = self.masks.transpose((0, 3, 1, 2))

        file_names = [f for f in os.listdir(data_dir) if f != '.DS_Store'] # ignore .DS store on Macs

        # self.images, self.masks, self.labels, self.planes, self.file_names = self.filter_data(self.images, 
        #                                                              self.masks, 
        #                                                              self.labels, 
        #                                                              self.planes, 
        #                                                              file_names)
        

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index) -> Any:

        self.masks[index] = torch.from_numpy(self.masks[index])
        self.images[index] = torch.from_numpy(self.images[index])
        self.images[index] = min_max_normalize(self.images[index])

        return self.images[index],  self.masks[index], self.labels[index]




if __name__ == '__main__':

    image_paths = glob.glob(os.path.join(BRAIN_DATA, '*'))[:1000]
    print(f"#IMAGES:{len(image_paths)}")

    dataset    = BrainDataset(data_dir=BRAIN_DATA)
    dataloader = DataLoader(dataset=dataset, batch_size=30)
    print(f"#BATCHES:{len(dataloader)}")
    data = next(iter(dataloader))

    for idx, batch in enumerate(dataloader):
        print(torch.min(batch[0]))
        print(torch.max(batch[0]))
        print(batch[0].shape, batch[1].shape, batch[2].shape)
        input()
