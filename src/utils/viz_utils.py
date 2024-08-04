import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os


def plot_results(imgs, recons, save_path, epoch, batch):

    bs = 8 #imgs.size(0)
    fig, axes = plt.subplots(nrows=2,ncols=bs,figsize=(bs,15))

    for i, (row,col) in enumerate(product(range(2),range(bs))):

        if row == 0:
            axes[row][col].imshow(np.transpose(imgs[col].detach().cpu().numpy(),(1,2,0)))
            if col == 0:
                axes[row][col].set_ylabel('Original Image',fontsize=15,fontweight='bold')
        
        elif row == 1:
            axes[row][col].imshow(np.transpose(recons[col].detach().cpu().numpy(),(1,2,0)))

            if col == 0:
                axes[row][col].set_ylabel('Reconstructed Image',fontsize=15,fontweight='bold')

            
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{batch}.jpg'),format='jpg',bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show() 
    plt.close()

def visualize_samples(tensor, save_path, epoch, batch):

    tensor = tensor.cpu().numpy()
    tensor = tensor.squeeze(1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(tensor[i])
        axes[i].axis('off')
    print("SaveDIR:",os.path.join(save_path,f'fig_{epoch}_{batch}.jpg'))
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{batch}.jpg'),format='jpg',bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show() 
    plt.close()

def save_tensor_as_jpg(tensor, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    length_dir = glob.glob(os.path.join(save_dir,'*'))

    for idx, img in enumerate(tensor):

        transform = transforms.ToPILImage()
        image = transform(img)
        image.save(os.path.join(save_dir,f'{idx+len(length_dir)}.jpeg'), 'JPEG')


def plot_from_dir(folder_path, num_sample=25):

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png') or f.lower().endswith('.jpeg')]
    
    if len(image_files) > num_sample:
        image_files = random.sample(image_files, num_sample)
    

    grid_size = int(num_sample ** 0.5)
    if grid_size ** 2 < num_sample:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for ax, img_file in zip(axes, image_files):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    for ax in axes[len(image_files):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('figure-gray.jpg',format='jpg',bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show()

def visualize_tensor(tensor):
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    tensor = tensor.numpy()
    tensor = tensor.squeeze(1)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(tensor[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

    