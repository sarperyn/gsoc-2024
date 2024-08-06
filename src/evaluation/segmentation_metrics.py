import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    
    return dice

def iou_coefficient(y_true, y_pred, smooth=1e-6):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou