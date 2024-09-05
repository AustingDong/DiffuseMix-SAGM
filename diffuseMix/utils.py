from diffuseMix.diffusion import Diffusion
import numpy as np
import random

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
        

        mat_u1 = np.zeros(shape=org_img_arr.shape)
        mat_u1[:org_img_arr.shape[0], :] = 1
        mat_u0 = ones_matrix - mat_u1

        mat_l1 = np.zeros(shape=org_img_arr.shape)
        mat_l1[:, :org_img_arr.shape[1]] = 1
        mat_l0 = ones_matrix - mat_l1

        masks = [mat_u0, mat_u1, mat_l0, mat_l1]

        mask = random(masks)

        print(org_img_arr.shape)
        print(gen_img_arr.shape)
        print(ones_matrix.shape)
        print(mask)

        return mask * org_img + (ones_matrix - mask) * gen_img
