import torch
import torchvision

def img_visualize(x, step, writer):
    # shape x: (3, 2, 32, 3, 224, 224)
    # self.check_shape(x)
    
    def min_max_normalize(standardized_image):
        min_val = standardized_image.min()
        max_val = standardized_image.max()

        if max_val != min_val:
            scaled_image = (standardized_image - min_val) / (max_val - min_val)
        else:
            # If all values are the same, set the image to zeros
            scaled_image = torch.zeros_like(standardized_image)
        return scaled_image
    
    if step % 100 == 0:
        for k in range(len(x)):
            x_c = x[k] # (2, 32, 3, 224, 224)
            x_original = x_c[0]   # (32, 3, 224, 224)
            x_augmented = x_c[1]  # (32, 3, 224, 224)

            for i in range(len(x_c)):
                original_image = x_original[i] # (3, 224, 224)
                transformed_image = x_augmented[i] # (3, 224, 224)
                
                # From visualization, the images are standardized, not min-max normalized
                # Apply min-max normalize to approximately scale to [0, 1], then scale back to [0, 255]
                original_image = min_max_normalize(original_image)
                transformed_image = min_max_normalize(transformed_image)

                # Keep everything on GPU
                original_image = original_image * 255  # Scale to [0, 255]
                transformed_image = transformed_image * 255

                # Ensure images are clamped to valid range
                original_image = original_image.clamp(0, 255).to(torch.uint8)  # Shape [3, 224, 224]
                transformed_image = transformed_image.clamp(0, 255).to(torch.uint8)

                # Stack images on GPU
                batch_images = torch.stack([original_image, transformed_image])  # Shape [2, 3, 224, 224]

                # Create a grid from the batch directly on GPU
                img_grid = torchvision.utils.make_grid(batch_images)

                # Write to TensorBoard (move the final grid to CPU only for visualization)
                writer.add_image('original_vs_transformed_image', img_grid.cpu(), global_step=i + len(x) * k + 100 * step)