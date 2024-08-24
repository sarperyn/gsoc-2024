import gradio as gr
from train_sample_ddim import *
from sample_vae import *


def generate_synthetic_data(model, dataset, samples, timesteps=50, eta=0):
    # Your logic to generate data using the selected model and parameters
    if model == "Diffusion":
        sample_ddim(mode='sample', dataset=dataset.upper(), n_samples=int(samples), timestep=int(timesteps), eta=float(eta))
        return f"Generating {samples} samples using {model} with timesteps {timesteps} and eta {eta} on {dataset} dataset."
    
    if model == "VAE":
        sample_with_vae(sample_size=samples)
        return f"Generating {samples} samples using {model} on {dataset} dataset."

    if model == "GAN":
        return f"Generating {samples} samples using {model} on {dataset} dataset."

def get_synthetic_data_generation_interface():
    with gr.Blocks() as synthetic_data_generation_interface:
        with gr.Row():
            model_choice = gr.Radio(["VAE", "Diffusion"], label="Choose Model")
        
        with gr.Row():
            dataset_choice = gr.Dropdown(["Madison"], label="Choose Dataset")

        with gr.Row():
            samples = gr.Slider(1, 100, value=10, step=1, label="Number of Samples to Generate")

        with gr.Row(visible=False) as diffusion_params:
            timesteps = gr.Slider(1, 1000, value=100, step=1, label="Timesteps (Diffusion)")
            eta = gr.Slider(0, 1, value=0.5, step=0.01, label="Eta (Diffusion)")

        def update_visibility(model):
            if model == "Diffusion":
                return gr.update(visible=True)
            if model == "VAE":
                return gr.update(visible=False)
            if model == "GAN":
                return gr.update(visible=False)

        model_choice.change(
            fn=update_visibility,
            inputs=model_choice,
            outputs=diffusion_params
        )

        generate_button = gr.Button("Generate")
        output = gr.Textbox(label="Output")

        def on_generate_click(model, dataset, samples, timesteps, eta):
            return generate_synthetic_data(model, dataset, samples, timesteps, eta)

        generate_button.click(
            fn=on_generate_click,
            inputs=[model_choice, dataset_choice, samples, timesteps, eta],
            outputs=output
        )

    return synthetic_data_generation_interface

if __name__ == "__main__":
    interface = get_synthetic_data_generation_interface()
    interface.launch()