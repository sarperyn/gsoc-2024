import numpy as np
import torch
import glob
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import DataLoader
import torch.optim as optim

from src.dataloader.dataloaders import MadisonDataset
from src.models.ddpm import DiffusionModel, UNet
from src.utils.viz_utils import visualize_samples
from src.utils.args_utils import arg_parser
from src.utils.variable_utils import MADISON_DATA, PLOT_DIRECTORY


args = arg_parser()
#init_wandb(args)

image_paths = glob.glob(os.path.join(MADISON_DATA, '*'))
print(f"#IMAGES:{len(image_paths)}")

dataset    = MadisonDataset(image_paths=image_paths)
dataloader = DataLoader(dataset=dataset, batch_size=args.bs)
device = args.device


unet = UNet(in_channels=1, out_channels=1).to(device)
model = DiffusionModel(unet, args).to(device)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def normalize(x):
    return 2 * x - 1

def denormalize(x):
    return (x + 1) / 2


def train_model(model, train_loader, device, args):

    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id),exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id,'model'),exist_ok=True)

    model.train()
    for epoch in range(args.epoch):
        train_loss = 0
        for batch_idx, x in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):

            x = x.to(device)
            x = normalize(x)
            #visualize_samples(tensor=x, save_path=os.path.join(PLOT_DIRECTORY,args.exp_id), epoch=epoch, batch=batch_idx)
            #print(torch.min(x), torch.max(x))
            t = torch.randint(0, model.num_timesteps, (x.size(0),), device=device).long()
            optimizer.zero_grad()
            loss = model.loss_function(x, t)
            loss.backward()
            optimizer.step()

            if batch_idx % 30 == 0:
               print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(x)}')

            if batch_idx % 100 == 0:
                sample_shape = (5, 1, 256, 256) 
                generated_samples = model.sample(sample_shape, device)
                print(generated_samples)
                torch.save(model, os.path.join(PLOT_DIRECTORY,args.exp_id,'model',f'model_{args.exp_id.split('/')[-1]}.pt'))
                visualize_samples(tensor=generated_samples, save_path=os.path.join(PLOT_DIRECTORY,args.exp_id), epoch=epoch, batch=batch_idx)

        print(f'Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}')

    print("Training completed.")

train_model(model=model, train_loader=dataloader, device=device, args=args)