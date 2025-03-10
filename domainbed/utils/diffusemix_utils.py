import random
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import math
import os
from time import time

class AdaptiveDiffuseMixUtils:
    @staticmethod
    @torch.no_grad()
    def create_image(original_img, augmented_img, fractal_root, num_slices, blend_width=20, alpha=0.20):
        # choose mask and blend images
        base_img_size = original_img.size
        if num_slices == 2:
            blended = AdaptiveDiffuseMixUtils.basic_blend(original_img, augmented_img, blend_width)
        else:
            blended = AdaptiveDiffuseMixUtils.blend_checkerboard(original_img, augmented_img, num_slices, blend_width)
        
        # choose a random fractal image from root
        fractal_img_paths = []
        fractal_img_paths = []
        for root, _, files in os.walk(fractal_root):
            for fname in files:
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    fractal_img_paths.append(os.path.join(root, fname))
        fractal_img_path = random.choice(fractal_img_paths)

        loader = Image.open
        fractal_img = loader(fractal_img_path)
        return AdaptiveDiffuseMixUtils.blend_images_with_resize(blended, fractal_img, alpha, base_img_size)

    @torch.no_grad()
    def blend_checkerboard(original_img, augmented_img, num_slices, blend_width=20):
        """
        Blend two square images in a checkerboard pattern with smooth transitions.
        
        Args:
            original_img (PIL.Image): First input image (must be square)
            augmented_img (PIL.Image): Second input image (must be square)
            blend_width (int): Width of the blending transition zone
            num_slices (int): Number of divisions (must be a power of 4)
        
        Returns:
            PIL.Image: Blended image
        """
        blend_width = (int)(blend_width)
        
        # Validate inputs
        if not (math.log(num_slices, 4).is_integer()):
            raise ValueError("num_slices must be a power of 4 (4, 16, 64, etc.)")
        
        size = original_img.size[0]  # Image is square, so we only need one dimension
        if original_img.size != (size, size) or augmented_img.size != (size, size):
            raise ValueError("Both images must be square and of the same size")
        
        grid_size = int(math.sqrt(num_slices))  # Convert to 2D grid size (2 for 4 slices, 4 for 16 slices, etc.)
        cell_size = size // grid_size
        
        # Create base mask array
        mask = np.zeros((size, size))
        
        # Fill the mask with checkerboard pattern
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 1:  # Checker pattern
                    top = i * cell_size
                    bottom = (i + 1) * cell_size
                    left = j * cell_size
                    right = (j + 1) * cell_size
                    mask[top:bottom, left:right] = 1
        
        # Apply blending at all internal borders
        half_blend = blend_width // 2
        
        # Generate transition arrays once
        h_transition = np.linspace(0, 1, blend_width)
        v_transition = h_transition.reshape(-1, 1)
        
        # Horizontal internal borders
        for i in range(1, grid_size):
            y_pos = i * cell_size
            # Create blend region
            mask[y_pos-half_blend:y_pos+half_blend, :] = (
                (1 - v_transition) * mask[y_pos-half_blend:y_pos+half_blend, :] +
                v_transition * np.flip(mask[y_pos-half_blend:y_pos+half_blend, :], axis=0)
            )
        
        # Vertical internal borders
        for j in range(1, grid_size):
            x_pos = j * cell_size
            # Create blend region
            mask[:, x_pos-half_blend:x_pos+half_blend] = (
                (1 - h_transition) * mask[:, x_pos-half_blend:x_pos+half_blend] +
                h_transition * np.flip(mask[:, x_pos-half_blend:x_pos+half_blend], axis=1)
            )
        
        # Convert PIL images to PyTorch tensors
        # original_tensor = torch.from_numpy(np.array(original_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        # augmented_tensor = torch.from_numpy(np.array(augmented_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        original_tensor = ToTensor()(original_img)
        augmented_tensor = ToTensor()(augmented_img)
        
        # Expand mask to handle RGB
        mask = torch.from_numpy(mask).float().unsqueeze(0).repeat(3, 1, 1)
        
        # Blend images
        blended_tensor = (1 - mask) * original_tensor + mask * augmented_tensor
        # blended_tensor = torch.clamp(blended_tensor * 255, 0, 255).byte()  # Scale back to [0, 255]
        
        return blended_tensor
    
    @staticmethod
    @torch.no_grad()
    def basic_blend(original_img, augmented_img, blend_width):
        width, height = original_img.size
        combine_choice = random.choice(['horizontal', 'vertical'])
        blend_width = (int)(blend_width)

        # Convert PIL images to PyTorch tensors
        # original_tensor = torch.from_numpy(np.array(original_img, dtype=np.float32) / 255.0).permute(2, 0, 1)  # (C, H, W)
        # augmented_tensor = torch.from_numpy(np.array(augmented_img, dtype=np.float32) / 255.0).permute(2, 0, 1)  # (C, H, W)
        original_tensor = ToTensor()(original_img)
        augmented_tensor = ToTensor()(augmented_img)

        # Initialize the blending mask
        if combine_choice == 'vertical':  # Vertical combination
            mask = torch.linspace(0, 1, blend_width).view(-1, 1).repeat(1, width)  # (H, W)
            mask = torch.cat([
                torch.zeros(height // 2 - blend_width // 2, width),
                mask,
                torch.ones(height // 2 - blend_width // 2 + blend_width % 2, width)
            ], dim=0)
        else:  # Horizontal combination
            mask = torch.linspace(0, 1, blend_width).view(1, -1).repeat(height, 1)  # (H, W)
            mask = torch.cat([
                torch.zeros(height, width // 2 - blend_width // 2),
                mask,
                torch.ones(height, width // 2 - blend_width // 2 + blend_width % 2)
            ], dim=1)

        # Expand the mask to match the image channels
        mask = mask.unsqueeze(0).repeat(3, 1, 1)  # (C, H, W)

        # Perform blending
        blended_tensor = (1 - mask) * original_tensor + mask * augmented_tensor
        # blended_tensor = torch.clamp(blended_tensor * 255, 0, 255).byte()  # Scale back to [0, 255]

        # mask = Image.fromarray((mask[0] * 255).byte().cpu().numpy())
        # mask.save("test/result/mask.jpg")
        return blended_tensor

    @staticmethod
    @torch.no_grad()
    def blend_images_with_resize(base_tensor, overlay_img, alpha=0.20, base_img_size=(224, 224)):
        overlay_img_resized = overlay_img.resize(base_img_size)
        # base_array = np.array(base_img, dtype=np.float32)
        # overlay_array = np.array(overlay_img_resized, dtype=np.float32)
        # assert base_array.shape == overlay_array.shape and len(base_array.shape) == 3
        # blended_array = (1 - alpha) * base_array + alpha * overlay_array
        # blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        # blended_img = Image.fromarray(blended_array)
        # return blended_img

        # overlay_tensor = torch.from_numpy(np.array(overlay_img_resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
        overlay_tensor = ToTensor()(overlay_img_resized)
        blended_tensor = (1 - alpha) * base_tensor + alpha * overlay_tensor
        blended_tensor = torch.clamp(blended_tensor * 255, 0, 255).byte()  # Scale back to [0, 255]
        blended_img = Image.fromarray(blended_tensor.permute(1, 2, 0).cpu().numpy())
        return blended_img


if __name__ == "__main__":
    original_img = Image.open("test/test_images/original/t1.jpg").resize((224, 224))
    augmented_img = Image.open("test/test_images/generated/t1.jpg").resize((224, 224))
    fractal_root = "test/test_images/fractal"
    num_slices = 2

    start_time = time()
    blended_img = AdaptiveDiffuseMixUtils.create_image(original_img, augmented_img, fractal_root, num_slices)
    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")
    # write the blended image to a file
    output_path = "test/result/blended.jpg"
    blended_img.save(output_path)