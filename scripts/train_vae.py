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
from src.models.vanilla_vae import ConvVAE
from src.utils.viz_utils import plot_results
from src.utils.args_utils import train_arg_parser
from src.utils.variable_utils import MADISON_DATA, PLOT_DIRECTORY


args = train_arg_parser()
#init_wandb(args)

image_paths = glob.glob(os.path.join(MADISON_DATA, '*'))
print(f"#IMAGES:{len(image_paths)}")

dataset    = MadisonDataset(image_paths=image_paths)
dataloader = DataLoader(dataset=dataset, batch_size=args.bs)
device = args.device

model = ConvVAE()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)



def train_model(model, train_loader, device, args):

    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id),exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id,'model'),exist_ok=True)

    model.train()
    for epoch in range(args.epoch):
        train_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):

            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 30 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(data)}')
                plot_results(imgs=data, recons=recon_batch, save_path=os.path.join(PLOT_DIRECTORY,args.exp_id), epoch=epoch, batch=batch_idx)
                torch.save(model, os.path.join(PLOT_DIRECTORY,args.exp_id,'model',f'model_{args.exp_id.split('/')[-1]}.pt'))
    
        print(f'Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}')

    print("Training completed.")


train_model(model=model, train_loader=dataloader, device=device, args=args)
