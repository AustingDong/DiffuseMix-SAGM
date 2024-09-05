import torch
from diffusers import StableDiffusionImg2ImgPipeline

class Diffusion:

    def __init__(self, model_id, seed=1145):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float16).to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def generate(self, prompt, image):
        
        gen_img = self.pipeline(
            prompt = prompt,
            negative_prompt="Oversaturated, blurry, low quality, strange eyes, strange hands", # What NOT to generate
            image=image,
            guidance_scale=8,          # How strongly to follow the prompt
            generator = self.generator 
        )

        return gen_img
