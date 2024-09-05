from diffuseMix.diffusion import Diffusion
import numpy as np

class Utils:
    
    @staticmethod
    def generate(prompt, image, model_id="nitrosocke/Ghibli-Diffusion"):
        diffuser = Diffusion(model_id=model_id, seed=1024)
        gen_img = diffuser.generate(prompt=prompt, image=image)
        return gen_img
    
    @staticmethod
    def combine(org_img, gen_img):
        org_img_arr = np.array(org_img)
        gen_img_arr = np.array(gen_img)

        ones_matrix = np.ones(shape=org_img_arr.shape)

        print(org_img_arr.shape)
        print(gen_img_arr.shape)
        print(ones_matrix.shape)


