import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, AutoPipelineForInpainting, TCDScheduler
from diffusers.utils import make_image_grid
from PIL import Image
import io
import requests

# Available models
AVAILABLE_MODELS = {
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Animagine XL 3.0": "cagliostrolab/animagine-xl-3.0",
}

# Available LoRA styles
AVAILABLE_LORAS = {
    "TCD": "h1t/TCD-SDXL-LoRA",
    "Papercut": "TheLastBen/Papercut_SDXL",
}

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def generate_image(prompt, seed, num_steps, guidance_scale, eta):
    # Initialize the pipeline
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"
    
    # Use CPU for inference
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    # Load and fuse LoRA weights
    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()
    
    # Generate the image
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        generator=generator,
    ).images[0]
    
    return image

def generate_community_image(prompt, model_name, seed, num_steps, guidance_scale, eta):
    # Initialize the pipeline
    base_model_id = AVAILABLE_MODELS[model_name]
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"
    
    # Use CPU for inference
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    # Load and fuse LoRA weights
    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()
    
    # Generate the image
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        generator=generator,
    ).images[0]
    
    return image

def generate_style_mix(prompt, seed, num_steps, guidance_scale, eta, style_weight):
    # Initialize the pipeline
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"
    styled_lora_id = "TheLastBen/Papercut_SDXL"
    
    # Use CPU for inference
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    # Load multiple LoRA weights
    pipe.load_lora_weights(tcd_lora_id, adapter_name="tcd")
    pipe.load_lora_weights(styled_lora_id, adapter_name="style")
    
    # Set adapter weights
    pipe.set_adapters(["tcd", "style"], adapter_weights=[1.0, style_weight])
    
    # Generate the image
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        generator=generator,
    ).images[0]
    
    return image

def inpaint_image(prompt, init_image, mask_image, seed, num_steps, guidance_scale, eta, strength):
    # Initialize the pipeline
    base_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"
    
    # Use CPU for inference
    pipe = AutoPipelineForInpainting.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    # Load and fuse LoRA weights
    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()
    
    # Generate the image
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        strength=strength,
        generator=generator,
    ).images[0]
    
    # Create a grid of the original image, mask, and result
    grid = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    return grid

# Create the Gradio interface
with gr.Blocks(title="TCD-SDXL Image Generator") as demo:
    gr.Markdown("# TCD-SDXL Image Generator")
    gr.Markdown("Generate images using Trajectory Consistency Distillation with Stable Diffusion XL. Note: This runs on CPU, so generation may take some time.")
    
    with gr.Tabs():
        with gr.TabItem("Text to Image"):
            with gr.Row():
                with gr.Column():
                    text_prompt = gr.Textbox(
                        label="Prompt",
                        value="Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna.",
                        lines=3
                    )
                    text_seed = gr.Slider(minimum=0, maximum=2147483647, value=0, label="Seed", step=1)
                    text_steps = gr.Slider(minimum=1, maximum=10, value=4, label="Number of Steps", step=1)
                    text_guidance = gr.Slider(minimum=0, maximum=1, value=0, label="Guidance Scale")
                    text_eta = gr.Slider(minimum=0, maximum=1, value=0.3, label="Eta")
                    text_button = gr.Button("Generate")
                with gr.Column():
                    text_output = gr.Image(label="Generated Image")
            
            text_button.click(
                fn=generate_image,
                inputs=[text_prompt, text_seed, text_steps, text_guidance, text_eta],
                outputs=text_output
            )
        
        with gr.TabItem("Inpainting"):
            with gr.Row():
                with gr.Column():
                    inpaint_prompt = gr.Textbox(
                        label="Prompt",
                        value="a tiger sitting on a park bench",
                        lines=3
                    )
                    init_image = gr.Image(label="Initial Image", type="pil")
                    mask_image = gr.Image(label="Mask Image", type="pil")
                    inpaint_seed = gr.Slider(minimum=0, maximum=2147483647, value=0, label="Seed", step=1)
                    inpaint_steps = gr.Slider(minimum=1, maximum=10, value=8, label="Number of Steps", step=1)
                    inpaint_guidance = gr.Slider(minimum=0, maximum=1, value=0, label="Guidance Scale")
                    inpaint_eta = gr.Slider(minimum=0, maximum=1, value=0.3, label="Eta")
                    inpaint_strength = gr.Slider(minimum=0, maximum=1, value=0.99, label="Strength")
                    inpaint_button = gr.Button("Inpaint")
                with gr.Column():
                    inpaint_output = gr.Image(label="Result (Original | Mask | Generated)")
            
            inpaint_button.click(
                fn=inpaint_image,
                inputs=[
                    inpaint_prompt, init_image, mask_image, inpaint_seed,
                    inpaint_steps, inpaint_guidance, inpaint_eta, inpaint_strength
                ],
                outputs=inpaint_output
            )
            
        with gr.TabItem("Community Models"):
            with gr.Row():
                with gr.Column():
                    community_prompt = gr.Textbox(
                        label="Prompt",
                        value="A man, clad in a meticulously tailored military uniform, stands with unwavering resolve. The uniform boasts intricate details, and his eyes gleam with determination. Strands of vibrant, windswept hair peek out from beneath the brim of his cap.",
                        lines=3
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(AVAILABLE_MODELS.keys()),
                        value="Animagine XL 3.0",
                        label="Select Model"
                    )
                    community_seed = gr.Slider(minimum=0, maximum=2147483647, value=0, label="Seed", step=1)
                    community_steps = gr.Slider(minimum=1, maximum=10, value=8, label="Number of Steps", step=1)
                    community_guidance = gr.Slider(minimum=0, maximum=1, value=0, label="Guidance Scale")
                    community_eta = gr.Slider(minimum=0, maximum=1, value=0.3, label="Eta")
                    community_button = gr.Button("Generate")
                with gr.Column():
                    community_output = gr.Image(label="Generated Image")
            
            community_button.click(
                fn=generate_community_image,
                inputs=[
                    community_prompt, model_dropdown, community_seed,
                    community_steps, community_guidance, community_eta
                ],
                outputs=community_output
            )
            
        with gr.TabItem("Style Mixing"):
            with gr.Row():
                with gr.Column():
                    style_prompt = gr.Textbox(
                        label="Prompt",
                        value="papercut of a winter mountain, snow",
                        lines=3
                    )
                    style_seed = gr.Slider(minimum=0, maximum=2147483647, value=0, label="Seed", step=1)
                    style_steps = gr.Slider(minimum=1, maximum=10, value=4, label="Number of Steps", step=1)
                    style_guidance = gr.Slider(minimum=0, maximum=1, value=0, label="Guidance Scale")
                    style_eta = gr.Slider(minimum=0, maximum=1, value=0.3, label="Eta")
                    style_weight = gr.Slider(minimum=0, maximum=2, value=1.0, label="Style Weight", step=0.1)
                    style_button = gr.Button("Generate")
                with gr.Column():
                    style_output = gr.Image(label="Generated Image")
            
            style_button.click(
                fn=generate_style_mix,
                inputs=[
                    style_prompt, style_seed, style_steps,
                    style_guidance, style_eta, style_weight
                ],
                outputs=style_output
            )

if __name__ == "__main__":
    demo.launch() 