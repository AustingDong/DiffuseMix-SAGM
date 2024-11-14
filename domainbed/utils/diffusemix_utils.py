import random
import numpy as np
from PIL import Image
import math
import os

class AdaptiveDiffuseMixUtils:
    @staticmethod
    def create_image(original_img, augmented_img, fractal_root, loader, num_slices, blend_width=20, alpha=0.20):
        # choose mask and blend images
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
        fractal_img = loader(fractal_img_path)
        return AdaptiveDiffuseMixUtils.blend_images_with_resize(blended, fractal_img, alpha)

    
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
        
        # Convert images to arrays and normalize
        original_array = np.array(original_img, dtype=np.float32) / 255.0
        augmented_array = np.array(augmented_img, dtype=np.float32) / 255.0
        
        # Expand mask to handle RGB
        mask = np.stack([mask] * 3, axis=2)
        
        # Blend images
        blended_array = (1 - mask) * original_array + mask * augmented_array
        blended_array = np.clip(blended_array * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended_array)
    
    @staticmethod
    def basic_blend(original_img, augmented_img, blend_width):
        width, height = original_img.size
        combine_choice = random.choice(['horizontal', 'vertical'])
        blend_width = (int)(blend_width)

        if combine_choice == 'vertical':  # Vertical combination
            mask = np.linspace(0, 1, blend_width).reshape(-1, 1)
            mask = np.tile(mask, (1, width))  # Extend mask horizontally
            mask = np.vstack([np.zeros((height // 2 - blend_width // 2, width)), mask,
                              np.ones((height // 2 - blend_width // 2 + blend_width % 2, width))])
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        else:
            mask = np.linspace(0, 1, blend_width).reshape(1, -1)
            mask = np.tile(mask, (height, 1))  # Extend mask vertically
            mask = np.hstack([np.zeros((height, width // 2 - blend_width // 2)), mask,
                              np.ones((height, width // 2 - blend_width // 2 + blend_width % 2))])
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        original_array = np.array(original_img, dtype=np.float32) / 255.0
        augmented_array = np.array(augmented_img, dtype=np.float32) / 255.0

        blended_array = (1 - mask) * original_array + mask * augmented_array
        blended_array = np.clip(blended_array * 255, 0, 255).astype(np.uint8)

        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def blend_images_with_resize(base_img, overlay_img, alpha=0.20):
        overlay_img_resized = overlay_img.resize(base_img.size)
        base_array = np.array(base_img, dtype=np.float32)
        overlay_array = np.array(overlay_img_resized, dtype=np.float32)
        assert base_array.shape == overlay_array.shape and len(base_array.shape) == 3
        blended_array = (1 - alpha) * base_array + alpha * overlay_array
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        blended_img = Image.fromarray(blended_array)
        return blended_img


if __name__ == "__main__":
    original_img = Image.open("/Users/ethan/Downloads/root/PACS_augmented/cartoon_augmented/original_resized/dog/pic_001.jpg")
    augmented_img = Image.open("/Users/ethan/Downloads/root/PACS_augmented/cartoon_augmented/generated/dog/pic_001.jpg_generated_art_painting.jpg")
    fractal_root = "/Users/ethan/Downloads/root/PACS_augmented/cartoon_augmented/fractal"
    num_slices = 16
    blended_img = AdaptiveDiffuseMixUtils.create_image(original_img, augmented_img, fractal_root, Image.open, num_slices)
    
    # write the blended image to a file
    output_path = "/Users/ethan/Downloads/blended.jpg"
    blended_img.save(output_path)