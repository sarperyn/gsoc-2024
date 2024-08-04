import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import matplotlib.pyplot as plt
from src.models.ddpm import DiffusionModel, UNet
from src.utils.args_utils import arg_parser


args = arg_parser()
device = args.device

unet = UNet(in_channels=1, out_channels=1).to(device)

diffusion_model = DiffusionModel(unet,args).to(device)
diffusion_model = torch.load('/home/syurtseven/gsoc/reports/diff_exp/0/model/model_0.pt')

sample_shape = (1, 256, 256)
num_samples = 5
generated_samples = generate_samples(diffusion_model, sample_shape, num_samples, device)

torch.save(generated_samples, 'generated_samples0.pt')