import os
import random
import numpy as np
from PIL import Image

class Utils:
    
    @staticmethod
    def combine(org_img, gen_img):
        org_img_arr = np.array(org_img)
        gen_img_arr = np.array(gen_img)

        assert org_img_arr.shape == gen_img_arr.shape

        ones_matrix = np.ones(shape=org_img_arr.shape)
        

        mat_u1 = np.zeros(shape=org_img_arr.shape)
        mat_u1[:org_img_arr.shape[0] // 2, :] = 1
        mat_u0 = ones_matrix - mat_u1

        mat_l1 = np.zeros(shape=org_img_arr.shape)
        mat_l1[:, :org_img_arr.shape[1] // 2] = 1
        mat_l0 = ones_matrix - mat_l1

        masks = [mat_u0, mat_u1, mat_l0, mat_l1]

        mask = random.choice(masks)

        combined_img = mask * org_img + (ones_matrix - mask) * gen_img

        return combined_img.astype(int)

    @staticmethod
    def fractal_mixing(combined_img, fractal_img, ld=0.2):
        combined_img_arr = np.array(combined_img)
        fractal_img_arr = np.array(fractal_img)

        assert combined_img_arr.shape == fractal_img_arr.shape

        return fractal_img_arr * ld + combined_img_arr * (1 - ld)

