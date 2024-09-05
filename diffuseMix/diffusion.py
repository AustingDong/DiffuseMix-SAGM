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
            negative_prompt="lowres, bad anatomy, bad hands, text, error, (missing fingers, extra digit, fewer digits:1.3), cropped, worst quality, low quality, normal quality, jpeg artifacts,signature, watermark, username, blurry, artist name, (worst quality, low quality, extra digits:1.3), detached collar,(juvenile:1.3), (simplistic:1.3), (lack of maturity:1.3), (unconvincing age:1.3), (cliche:1.3), Poor quality image, (low resolution:1.3), (pixelated:1.3), (blurred:1.3), (grainy:1.3), (noisy:1.3), (compression artifacts:1.3), (over-exposure:1.3), (under-exposure:1.3), (bad lighting:1.3), (poor composition:1.3), (unattractive:1.3), (unappealing:1.3), (unedited:1.3), (amateur:1.3),(three arms:1.3)(poorly drawn hands:1.3),",
            image=image,
            guidance_scale=8,
            generator = self.generator 
        )

        return gen_img
