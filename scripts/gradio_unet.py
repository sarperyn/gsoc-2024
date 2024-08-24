import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.getcwd()))
from src.models.unet import BaseUNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    model_path = "/home/syurtseven/gsoc-2024/reports/seg/2/model/model_20.pt"
    model  = BaseUNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model  = model.to(DEVICE)
    model.eval()
    return model

def read_image_as_grayscale(image):
    image_np = np.array(image)
    grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(grayscale_image)

def preprocess_image(image):
    grayscale_image = read_image_as_grayscale(image)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(grayscale_image).unsqueeze(0).to(DEVICE)

def postprocess_output(output, original_image):
    output = torch.sigmoid(output)
    output = output.squeeze().cpu().numpy()
    mask = (output > 0.5).astype('uint8') * 255

    # Convert the mask to a PIL Image and resize it to match the original image size
    mask = Image.fromarray(mask)
    mask = mask.resize(original_image.size, resample=Image.BILINEAR)

    # Convert original image to RGBA
    original_image = original_image.convert("RGBA")

    # Create an RGBA version of the mask
    mask_rgba = Image.new("RGBA", original_image.size)
    mask_rgba.paste((255, 0, 0, 128), (0, 0), mask)  # Red color with alpha

    # Overlay the mask on the original image
    overlay_image = Image.alpha_composite(original_image, mask_rgba)

    return overlay_image

def predict(image):
    fixed_size = (512, 512)
    image = image.resize(fixed_size)

    model = load_model()
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = postprocess_output(output_tensor[0], image)
    output_image = output_image.resize((256, 256))

    return output_image

def get_image_segmentation_interface():

    image_segmentation_interface = gr.Interface(
        fn=predict,
        inputs=gr.components.Image(type="pil"),
        outputs=gr.components.Image(type="pil"),
        title="U-Net Image Segmentation",
        description="Upload an image to generate its segmentation map using a pre-trained U-Net model."
    )
    return image_segmentation_interface

# Example to launch the interface
if __name__ == "__main__":
    interface = get_image_segmentation_interface()
    interface.launch()
