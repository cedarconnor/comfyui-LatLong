import os
import torch
import numpy as np
from typing import Tuple, Optional

import folder_paths
from comfy.utils import ProgressBar

from .modules.equirectangular_processor import EquirectangularProcessor


custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))


class EquirectangularRotate:
    """ComfyUI node for rotating equirectangular images and adjusting horizon"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "yaw_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "pitch_rotation": ("FLOAT", {"default": -65.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "roll_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "horizon_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rotated_image",)
    FUNCTION = "rotate_equirectangular"
    CATEGORY = "LatLong"
    
    def rotate_equirectangular(self,
                             image: torch.Tensor,
                             yaw_rotation: float = 0.0,
                             pitch_rotation: float = -65.0,
                             roll_rotation: float = 0.0,
                             horizon_offset: float = 0.0,
                             interpolation: str = "lanczos") -> Tuple[torch.Tensor]:
        
        # Convert ComfyUI tensor format (B, H, W, C) to numpy
        batch_size = image.shape[0]
        processed_images = []
        
        # Progress bar for batch processing
        pbar = ProgressBar(batch_size)
        
        for i in range(batch_size):
            # Get single image and convert to numpy
            img_tensor = image[i]  # (H, W, C)
            img_numpy = img_tensor.cpu().numpy()
            
            # Keep float32 for processing to avoid value clipping
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)
            
            # Process the image
            processed_img = EquirectangularProcessor.rotate_equirectangular(
                img_numpy,
                yaw=yaw_rotation,
                pitch=pitch_rotation,
                roll=roll_rotation,
                horizon_offset=horizon_offset,
                interpolation=interpolation
            )
            
            # Ensure output is float32 in [0,1] range
            processed_img = np.clip(processed_img, 0.0, 1.0).astype(np.float32)
            
            processed_tensor = torch.from_numpy(processed_img)
            processed_images.append(processed_tensor)
            
            pbar.update(i + 1)
        
        # Stack back to batch format
        result = torch.stack(processed_images, dim=0)
        return (result,)


class EquirectangularCrop180:
    """ComfyUI node for cropping equirectangular images to 180 degrees"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "output_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "crop_to_180"
    CATEGORY = "LatLong"
    
    def crop_to_180(self,
                   image: torch.Tensor,
                   output_width: int = 1024,
                   output_height: int = 512,
                   maintain_aspect: bool = True,
                   interpolation: str = "lanczos") -> Tuple[torch.Tensor]:
        
        batch_size = image.shape[0]
        processed_images = []
        
        pbar = ProgressBar(batch_size)
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_numpy = img_tensor.cpu().numpy()
            
            # Keep float32 for processing
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)
            
            # Determine output dimensions
            if maintain_aspect:
                # For 180-degree crop, ideal aspect ratio is 2:1
                output_width = output_width
                output_height = output_width // 2
            
            # Crop the image
            cropped_img = EquirectangularProcessor.crop_to_180(
                img_numpy,
                output_width=output_width,
                output_height=output_height,
                interpolation=interpolation
            )
            
            # Ensure output is float32 in [0,1] range
            cropped_img = np.clip(cropped_img, 0.0, 1.0).astype(np.float32)
            
            cropped_tensor = torch.from_numpy(cropped_img)
            processed_images.append(cropped_tensor)
            
            pbar.update(i + 1)
        
        result = torch.stack(processed_images, dim=0)
        return (result,)


class EquirectangularProcessor_Combined:
    """Combined ComfyUI node for all equirectangular operations"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "yaw_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "pitch_rotation": ("FLOAT", {"default": -65.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "roll_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "horizon_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "crop_to_180": ("BOOLEAN", {"default": False}),
                "crop_to_square": ("BOOLEAN", {"default": False}),
                "output_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "process_equirectangular"
    CATEGORY = "LatLong"
    
    def process_equirectangular(self,
                              image: torch.Tensor,
                              yaw_rotation: float = 0.0,
                              pitch_rotation: float = -65.0,
                              roll_rotation: float = 0.0,
                              horizon_offset: float = 0.0,
                              crop_to_180: bool = False,
                              crop_to_square: bool = False,
                              output_width: int = 1024,
                              output_height: int = 512,
                              interpolation: str = "lanczos") -> Tuple[torch.Tensor]:
        
        batch_size = image.shape[0]
        processed_images = []
        
        pbar = ProgressBar(batch_size)
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_numpy = img_tensor.cpu().numpy()
            
            # Keep float32 for processing
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)
            
            # Process the image with all operations
            processed_img = EquirectangularProcessor.process_equirectangular(
                img_numpy,
                yaw=yaw_rotation,
                pitch=pitch_rotation,
                roll=roll_rotation,
                horizon_offset=horizon_offset,
                crop_to_180=crop_to_180,
                crop_to_square=crop_to_square,
                output_width=output_width if crop_to_180 else None,
                output_height=output_height if crop_to_180 else None,
                interpolation=interpolation
            )
            
            # Ensure output is float32 in [0,1] range
            processed_img = np.clip(processed_img, 0.0, 1.0).astype(np.float32)
            
            processed_tensor = torch.from_numpy(processed_img)
            processed_images.append(processed_tensor)
            
            pbar.update(i + 1)
        
        result = torch.stack(processed_images, dim=0)
        return (result,)


class EquirectangularCropSquare:
    """ComfyUI node for cropping equirectangular images to square (width = height)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("square_image",)
    FUNCTION = "crop_to_square"
    CATEGORY = "LatLong"
    
    def crop_to_square(self,
                      image: torch.Tensor,
                      interpolation: str = "lanczos") -> Tuple[torch.Tensor]:
        
        batch_size = image.shape[0]
        processed_images = []
        
        pbar = ProgressBar(batch_size)
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_numpy = img_tensor.cpu().numpy()
            
            # Keep float32 for processing
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)
            
            # Crop the image to square
            cropped_img = EquirectangularProcessor.crop_to_square(
                img_numpy,
                interpolation=interpolation
            )
            
            # Ensure output is float32 in [0,1] range
            cropped_img = np.clip(cropped_img, 0.0, 1.0).astype(np.float32)
            
            cropped_tensor = torch.from_numpy(cropped_img)
            processed_images.append(cropped_tensor)
            
            pbar.update(i + 1)
        
        result = torch.stack(processed_images, dim=0)
        return (result,)