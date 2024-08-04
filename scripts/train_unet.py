import numpy as np
import torch
import glob
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


from src.dataloader.unet_dataloader import BrainDataset
from src.models.unet import UNet
from src.utils.viz_utils import plot_results
from src.utils.args_utils import train_arg_parser
from src.utils.variable_utils import BRAIN_DATA, PLOT_DIRECTORY


args = train_arg_parser()

image_paths = glob.glob(os.path.join(BRAIN_DATA, '*'))
print(f"#IMAGES:{len(image_paths)}")

dataset    = BrainDataset(data_dir=image_paths)
dataloader = DataLoader(dataset=dataset, batch_size=args.bs)
device = args.device

# Define the model
model = UNet(n_channels=1, n_classes=3)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def train_model(model, train_loader, device, args):

    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id),exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id,'model'),exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epoch):
        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):
            
            images, masks, labels = batch
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 30 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(images)}')
                plot_results(imgs=images, recons=outputs, save_path=os.path.join(PLOT_DIRECTORY,args.exp_id), epoch=epoch, batch=batch_idx)
                torch.save(model, os.path.join(PLOT_DIRECTORY,args.exp_id,'model',f'model_{args.exp_id.split('/')[-1]}.pt'))
    
        print(f'Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}')

    print("Training completed.")

train_model(model=model, train_loader=dataloader, device=device, args=args)