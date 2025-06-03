import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, TCDScheduler

def generate_image(prompt, seed, num_steps, guidance_scale, eta):
    # Initialize the pipeline
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"
    
    # Use CPU for inference
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
        variant="fp16"
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

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Prompt",
            value="Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna.",
            lines=3
        ),
        gr.Slider(minimum=0, maximum=2147483647, value=0, label="Seed", step=1),
        gr.Slider(minimum=1, maximum=10, value=4, label="Number of Steps", step=1),
        gr.Slider(minimum=0, maximum=1, value=0, label="Guidance Scale"),
        gr.Slider(minimum=0, maximum=1, value=0.3, label="Eta"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="TCD-SDXL Image Generator",
    description="Generate images using Trajectory Consistency Distillation with Stable Diffusion XL. Note: This runs on CPU, so generation may take some time.",
)

if __name__ == "__main__":
    demo.launch() 