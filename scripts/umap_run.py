from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import matplotlib.pyplot as plt
from src.utils.viz_utils import plot_from_dir
from src.dataloader.unet_dataloader import BrainDataset
from torch.utils.data import DataLoader
from src.utils.variable_utils import BRAIN_DATA


dataset    = BrainDataset(data_dir=BRAIN_DATA)
dataloader = DataLoader(dataset=dataset, batch_size=2034)
print(f"#BATCHES:{len(dataloader)}")
data = next(iter(dataloader))

brain_imgs_flatten = data[0].reshape(2034,-1)

brain_embeddings = umap.UMAP(random_state=42).fit_transform(brain_imgs_flatten)
np.save('brain_embeddings.npy', brain_embeddings)
tumor_type_to_label = {'meningioma': 1, 'glioma': 2, 'pituitary': 3}
label_to_tumor_type = {v: k for k, v in tumor_type_to_label.items()}

fig, ax = plt.subplots(figsize=(10, 8))

colors = {1: 'red', 2: 'blue', 3: 'green'}

for label, color in colors.items():
    indices = data[2] == label
    ax.scatter(brain_embeddings[indices, 0], brain_embeddings[indices, 1], c=color, s=2.5, label=label_to_tumor_type[label])

ax.set_title('UMAP Embeddings of Brain Data', fontsize=16)
ax.set_xlabel('UMAP Dimension 1', fontsize=14)
ax.set_ylabel('UMAP Dimension 2', fontsize=14)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(title='Tumor Type', fontsize=12, title_fontsize=14, markerscale=10)

plt.savefig('umap_brain.jpg', dpi=300, bbox_inches='tight')
plt.show()
