import numpy as np
import torch
import glob
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split


from src.dataloader.dataloaders import DatasetLabeled
from src.models.unet import BaseUNet
from src.utils.viz_utils import visualize_predictions, plot_metric
from src.utils.args_utils import train_arg_parser
from src.utils.variable_utils import PLOT_DIRECTORY, SEGMENTATION_DATA
from src.evaluation.segmentation_metrics import dice_coefficient


args = train_arg_parser()

device = args.device
dataset = DatasetLabeled(SEGMENTATION_DATA, args=args, augment=True)
#val_dataset   = MadisonDatasetLabeled(VALIDATION_DIR, augment=False)

train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=args.test_size, random_state=42)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# print(f"LENGTH TRAIN:{len(train_dataset)}")
# print(f"LENGTH VAL:{len(val_dataset)}")


val_dataset.dataset.augment = False

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

model = BaseUNet(in_channels=1, out_channels=1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, args):

    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id),exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIRECTORY,args.exp_id,'model'),exist_ok=True)

    train_loss_history = []
    val_loss_history   = []
    val_images = []
    dice_coef_history  = []
    iou_score_history  = []

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):
            images, masks, paths = batch
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            dice_score = dice_coefficient(masks, outputs)
            # iou_score = iou_coefficient(masks, outputs)
            
            train_loss += loss.item() * images.size(0)

        
        save_path = os.path.join(PLOT_DIRECTORY, args.exp_id, 'model', f'model_{epoch}.pt')
        torch.save(model.state_dict(), save_path)

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        dice_coefficients = 0.0
        # iou_coefficients  = 0.0 
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):

                images, masks, paths = batch
                images, masks = images.to(device), masks.to(device)
    
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                if epoch == 0:
                    for path in paths:
                        val_images.append(path)

                if batch_idx % 500 == 0:
                    visualize_predictions(images, masks, outputs, 
                                        save_path=os.path.join(PLOT_DIRECTORY,args.exp_id), 
                                        epoch=epoch, 
                                        batch_idx=batch_idx)
                    
                dice_score = dice_coefficient(masks, outputs)
                # iou_score = iou_coefficient(masks, outputs)

                dice_coefficients += dice_score
                # iou_coefficients  += iou_score


        dice_coefficients = dice_coefficients / len(val_loader.dataset)
        dice_coef_history.append(dice_coefficients)
        # iou_coefficients = iou_coefficients / len(val_loader.dataset)
        # iou_score_history.append(iou_coefficients)

        val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"Dice Coefficient for {epoch}: {dice_coefficients}")
        # print(f"IoU Coefficient for {epoch}: {iou_coefficients}")
        print(f'Epoch {epoch+1}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIRECTORY,args.exp_id,'train_loss_curves.jpg'))
    plt.show()
    plt.close()

    plot_metric(x=dice_coef_history, 
                label="dice coeff",
                plot_dir=PLOT_DIRECTORY,
                args=args,
                metric='dice_coeff')

    with open(os.path.join(PLOT_DIRECTORY, args.exp_id, 'all_image_paths.txt'), 'w') as f:
        for path in val_images:
            f.writelines(path)

    print("Training completed.")

train_model(model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            args=args)