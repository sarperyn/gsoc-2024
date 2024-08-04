import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import matplotlib.pyplot as plt
from src.models.vanilla_vae import ConvVAE
from src.utils.model_utils import set_model, sample_from_vae
from src.utils.args_utils import test_arg_parser
from src.utils.viz_utils import save_tensor_as_jpg



args = test_arg_parser()
model = set_model(ConvVAE, args.model_path, args.device)
generated_samples = sample_from_vae(model, args.sample_size, args.device, latent_dim=20)
save_tensor_as_jpg(generated_samples, args.save_dir)