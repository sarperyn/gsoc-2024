import gradio as gr
from gradio_unet import get_image_segmentation_interface
from gradio_data_gen import get_synthetic_data_generation_interface
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))


image_segmentation_interface        = get_image_segmentation_interface()
synthetic_data_generation_interface = get_synthetic_data_generation_interface()

demo = gr.TabbedInterface(
    interface_list=[
        image_segmentation_interface,
        synthetic_data_generation_interface
    ],
    tab_names=["Image Segmentation", "Synthetic Data Generation"]
)

demo.launch()
