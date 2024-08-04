import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
import torch.nn.functional as F
import cv2
import glob
import os
import umap
import numpy as np

_ = torch.manual_seed(123)


def get_generated_imgs(imgs_dir):

    img_dirs = glob.glob(os.path.join(imgs_dir,'*'))
    torch_list = []
    for img in img_dirs[:1000]:
        data = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        torch_list.append(data)

    torch_list = np.array(torch_list)
    generated_stack = torch.from_numpy(torch_list)

    return generated_stack

def get_real_imgs(imgs_dir):
        
    img_dirs = glob.glob(os.path.join(imgs_dir,'*'))

    torch_list = []
    for img in img_dirs[:1500]:
        data = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if data.shape == (266,266):
            torch_list.append(data)

    torch_list = torch_list[:1000]
    torch_list = np.array(torch_list)
    real_stack = torch.from_numpy(torch_list)

    return real_stack


generated_imgs_vae = get_generated_imgs(imgs_dir="/home/syurtseven/GI-Tract-Image-Segmentation/synthetic_data_generation/gsoc/scripts/results/vae")
print("GOT IT 1", generated_imgs_vae.shape)

generated_imgs_diff = get_generated_imgs(imgs_dir="/home/syurtseven/GI-Tract-Image-Segmentation/synthetic_data_generation/gsoc/scripts/results/diffusion")
print("GOT IT 1",generated_imgs_diff.shape)

real_imgs = get_real_imgs(imgs_dir="/home/syurtseven/GI-Tract-Image-Segmentation/synthetic_data_generation/gsoc/data/madison")
print("GOT IT 1", real_imgs.shape)

generated_imgs_vae = generated_imgs_vae.reshape(1000, 1, 266, 266)
generated_imgs_diff = generated_imgs_diff.reshape(1000, 1, 256, 256)
real_imgs = real_imgs.reshape(1000, 1, 266, 266)

# print("GOT IT 2", generated_imgs_vae.shape)
# print("GOT IT 2",generated_imgs_diff.shape)
# print("GOT IT 2", real_imgs.shape)


# def preprocess_images(images):
#     if images.dtype != torch.float32:
#         images = images.to(torch.float32)
#     return F.interpolate(images, size=(256, 256), mode='nearest')

# generated_imgs_vae_resized = preprocess_images(generated_imgs_vae)
# #generated_imgs_diff_resized = preprocess_images(generated_imgs_diff)
# real_imgs_resized = preprocess_images(real_imgs)

print(generated_imgs_vae.shape)
print(generated_imgs_diff.shape)
print(real_imgs.shape)

generated_imgs_vae_flatten = generated_imgs_vae.reshape(1000, -1)
generated_imgs_diff_flatten = generated_imgs_diff.reshape(1000, -1)
real_imgs_flatten = real_imgs.reshape(1000, -1)

generated_imgs_vae_embds  = umap.UMAP( n_neighbors=30,
                                    min_dist=0.0,
                                    n_components=2,
                                    random_state=42,
                                    ).fit_transform(generated_imgs_vae_flatten)
np.save('diffs.npy',generated_imgs_vae_embds)
generated_imgs_diff_embds = umap.UMAP( n_neighbors=30,
                                    min_dist=0.0,
                                    n_components=2,
                                    random_state=42,
                                    ).fit_transform(generated_imgs_diff_flatten)
np.save('vaess.npy',generated_imgs_diff_embds)
real_imgs_embds           = umap.UMAP( n_neighbors=30,
                                    min_dist=0.0,
                                    n_components=2,
                                    random_state=42,
                                    ).fit_transform(real_imgs_flatten)
np.save('reals.npy',real_imgs_embds)





# generated_imgs = generated_imgs.view((generated_imgs.shape[0], 1 , generated_imgs.shape[1], generated_imgs.shape[2]))
# print(generated_imgs.shape)
# input()
# fid.update(generated_imgs, real=False)
# fid.update(real_imgs, real=False)
# print(fid.compute())