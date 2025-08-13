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
    # Displayed by some UIs as a node tooltip/description
    DESCRIPTION = "Rotate an equirectangular (360°) image with yaw/pitch/roll and adjust the horizon."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "yaw_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Yaw: horizontal rotation around the vertical axis (degrees)."}),
                "pitch_rotation": ("FLOAT", {"default": -65.0, "min": -90.0, "max": 90.0, "step": 0.1, "tooltip": "Pitch: tilt up/down around the horizontal axis (degrees)."}),
                "roll_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Roll: bank left/right around the view axis (degrees)."}),
                "horizon_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1, "tooltip": "Vertical horizon shift (degrees). Positive values move the horizon up."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling filter used during rotation remap."}),
                "backend": (["auto", "cpu", "gpu"], {"default": "auto", "tooltip": "Processing backend. Auto uses GPU if available (bilinear/nearest)."}),
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
                             interpolation: str = "lanczos",
                             backend: str = "auto") -> Tuple[torch.Tensor]:
        
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
            
            # Choose backend
            use_gpu = (backend == 'gpu') or (backend == 'auto' and torch.cuda.is_available())
            if use_gpu:
                img_dev = image[i].to('cuda')
                proc_t = EquirectangularProcessor.torch_rotate_equirectangular(
                    img_dev,
                    yaw=yaw_rotation,
                    pitch=pitch_rotation,
                    roll=roll_rotation,
                    horizon_offset=horizon_offset,
                    interpolation=interpolation if interpolation in ('bilinear', 'nearest') else 'bilinear'
                ).to('cpu').numpy()
                processed_img = proc_t
            else:
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
    DESCRIPTION = "Extract a horizontal FOV (default 180°) window from an equirectangular image with optional aspect preservation."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "output_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1, "tooltip": "Target width of the extracted window in pixels."}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1, "tooltip": "Target height of the extracted window in pixels (overridden if Maintain Aspect)."}),
                "maintain_aspect": ("BOOLEAN", {"default": True, "tooltip": "Preserve the natural aspect ratio of the selected FOV by adjusting height."}),
                "center_longitude_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Longitude center of the crop in degrees (0° is center)."}),
                "fov_degrees": ("FLOAT", {"default": 180.0, "min": 1.0, "max": 360.0, "step": 1.0, "tooltip": "Horizontal field of view to extract (degrees)."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling filter for resizing after extraction."}),
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
                   center_longitude_deg: float = 0.0,
                   fov_degrees: float = 180.0,
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
            out_w = output_width
            out_h = output_height
            if maintain_aspect:
                # Maintain the crop's native aspect ratio (crop_width : height)
                H, W = img_numpy.shape[:2]
                crop_width_px = max(1, int(round(W * (float(fov_degrees) / 360.0))))
                out_h = max(1, int(round(out_w * (H / crop_width_px))))
            
            # Crop the image
            cropped_img = EquirectangularProcessor.crop_to_180(
                img_numpy,
                output_width=out_w,
                output_height=out_h,
                interpolation=interpolation,
                center_longitude_deg=center_longitude_deg,
                fov_degrees=fov_degrees
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
    DESCRIPTION = "Rotate an equirectangular image, then optionally crop to 180° or to a centered square — all in one node. (Square crop takes precedence.)"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                # Embedded sliders/toggles in the node UI
                "yaw_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Yaw: horizontal rotation around the vertical axis (degrees)."}),
                "pitch_rotation": ("FLOAT", {"default": -65.0, "min": -90.0, "max": 90.0, "step": 0.1, "tooltip": "Pitch: tilt up/down around the horizontal axis (degrees)."}),
                "roll_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Roll: bank left/right around the view axis (degrees)."}),
                "horizon_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1, "tooltip": "Vertical horizon shift (degrees). Positive values move the horizon up."}),
                "crop_to_180": ("BOOLEAN", {"default": False, "tooltip": "Enable crop to a horizontal FOV (usually 180°)."}),
                "crop_to_square": ("BOOLEAN", {"default": False, "tooltip": "Center-crop width to the image height; overrides 180° crop if enabled."}),
                "output_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1, "tooltip": "Output width for the 180° crop."}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1, "tooltip": "Output height for the 180° crop."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling filter used by rotation/crop steps."}),
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
    DESCRIPTION = "Center-crop an equirectangular image to a perfect square (width equals original height)."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Not used for pure center-crop; included for consistency."}),
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


class EquirectangularPerspectiveExtract:
    """ComfyUI node to extract rectilinear perspective views from equirectangular images"""
    DESCRIPTION = "Extract a pinhole-camera perspective view at a given yaw/pitch/roll and FOV from an equirectangular panorama."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "yaw_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Camera yaw (degrees)."}),
                "pitch_rotation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1, "tooltip": "Camera pitch (degrees)."}),
                "roll_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Camera roll (degrees)."}),
                "fov_degrees": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 179.0, "step": 0.5, "tooltip": "Horizontal field of view for the perspective camera (degrees)."}),
                "output_width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 1, "tooltip": "Output width of the perspective view in pixels."}),
                "output_height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 1, "tooltip": "Output height of the perspective view in pixels."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling filter used during perspective remap."}),
                "backend": (["auto", "cpu", "gpu"], {"default": "auto", "tooltip": "Processing backend. Auto uses GPU if available (bilinear/nearest)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("perspective_image",)
    FUNCTION = "extract_perspective"
    CATEGORY = "LatLong"

    def extract_perspective(self,
                            image: torch.Tensor,
                            yaw_rotation: float = 0.0,
                            pitch_rotation: float = 0.0,
                            roll_rotation: float = 0.0,
                            fov_degrees: float = 90.0,
                            output_width: int = 1024,
                            output_height: int = 1024,
                            interpolation: str = "lanczos",
                            backend: str = "auto") -> Tuple[torch.Tensor]:
        batch_size = image.shape[0]
        processed_images = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_tensor = image[i]
            img_numpy = img_tensor.cpu().numpy()
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)

            use_gpu = (backend == 'gpu') or (backend == 'auto' and torch.cuda.is_available())
            if use_gpu:
                img_dev = image[i].to('cuda')
                persp_t = EquirectangularProcessor.torch_perspective_extract(
                    img_dev,
                    out_width=output_width,
                    out_height=output_height,
                    yaw=yaw_rotation,
                    pitch=pitch_rotation,
                    roll=roll_rotation,
                    fov_degrees=fov_degrees,
                    interpolation=interpolation if interpolation in ('bilinear', 'nearest') else 'bilinear'
                ).to('cpu').numpy()
                persp_img = persp_t
            else:
                persp_img = EquirectangularProcessor.perspective_extract(
                    img_numpy,
                    out_width=output_width,
                    out_height=output_height,
                    yaw=yaw_rotation,
                    pitch=pitch_rotation,
                    roll=roll_rotation,
                    fov_degrees=fov_degrees,
                    interpolation=interpolation
                )

            persp_img = np.clip(persp_img, 0.0, 1.0).astype(np.float32)
            processed_images.append(torch.from_numpy(persp_img))
            pbar.update(i + 1)

        result = torch.stack(processed_images, dim=0)
        return (result,)


class EquirectangularToCubemap:
    """ComfyUI node to convert an equirectangular image into a 3x2 cubemap atlas."""
    DESCRIPTION = "Convert an equirectangular panorama into a 3×2 cubemap atlas (left, front, right / back, top, bottom)."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "face_size": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1, "tooltip": "Resolution (pixels) per cube face."}),
                "layout": (["3x2"], {"default": "3x2", "tooltip": "Output atlas layout (currently 3x2)."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling filter used during face generation."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cubemap_atlas",)
    FUNCTION = "to_cubemap"
    CATEGORY = "LatLong"

    def to_cubemap(self,
                   image: torch.Tensor,
                   face_size: int = 512,
                   layout: str = "3x2",
                   interpolation: str = "lanczos") -> Tuple[torch.Tensor]:
        batch_size = image.shape[0]
        processed_images = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_numpy = image[i].cpu().numpy()
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)
            atlas = EquirectangularProcessor.equirectangular_to_cubemap(
                img_numpy,
                face_size=face_size,
                layout=layout,
                interpolation=interpolation,
            )
            atlas = np.clip(atlas, 0.0, 1.0).astype(np.float32)
            processed_images.append(torch.from_numpy(atlas))
            pbar.update(i + 1)

        result = torch.stack(processed_images, dim=0)
        return (result,)
