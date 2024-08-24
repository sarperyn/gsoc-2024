import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import matplotlib.pyplot as plt
from src.models.vanilla_vae import ConvVAE
from src.utils.model_utils import set_model, sample_from_vae
from src.utils.args_utils import test_arg_parser
from src.utils.viz_utils import save_tensor_as_jpg



#args = test_arg_parser()

def sample_with_vae(sample_size=50):
    model_path = '/home/syurtseven/gsoc-2024/models/vae/vae.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = set_model(ConvVAE, model_path, device)
    generated_samples = sample_from_vae(model, sample_size, device, latent_dim=20)
    save_path = '/home/syurtseven/gsoc-2024/models/results/samples_vae'
    os.makedirs(save_path, exist_ok=True)
    save_tensor_as_jpg(generated_samples, save_path)