import os
import torch
import numpy as np
from typing import Tuple, Optional, Dict
import base64
from io import BytesIO

import folder_paths
from comfy.utils import ProgressBar
from PIL import Image

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


class CubemapToEquirectangular:
    """ComfyUI node to convert a 3x2 cubemap atlas back to equirectangular format."""
    DESCRIPTION = "Convert a 3×2 cubemap atlas (left, front, right / back, top, bottom) back to equirectangular panorama."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cubemap_atlas": ("IMAGE", {"tooltip": "Input cubemap atlas in 3×2 layout from 'Equirectangular → Cubemap' node."}),
            },
            "optional": {
                "output_width": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 1, "tooltip": "Width of output equirectangular panorama (typically 2× height for standard 2:1 ratio)."}),
                "output_height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 1, "tooltip": "Height of output equirectangular panorama."}),
                "layout": (["3x2"], {"default": "3x2", "tooltip": "Cubemap layout format (currently only 3×2 supported)."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling quality: lanczos (highest), bicubic, bilinear, nearest (fastest)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("equirectangular_image",)
    FUNCTION = "to_equirectangular"
    CATEGORY = "LatLong"

    def to_equirectangular(self,
                          cubemap_atlas: torch.Tensor,
                          output_width: int = 2048,
                          output_height: int = 1024,
                          layout: str = "3x2",
                          interpolation: str = "lanczos") -> Tuple[torch.Tensor]:
        batch_size = cubemap_atlas.shape[0]
        processed_images = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            atlas_numpy = cubemap_atlas[i].cpu().numpy()
            if atlas_numpy.dtype != np.float32:
                atlas_numpy = atlas_numpy.astype(np.float32)

            equirect = EquirectangularProcessor.cubemap_to_equirectangular(
                atlas_numpy,
                output_width=output_width,
                output_height=output_height,
                layout=layout,
                interpolation=interpolation,
            )
            equirect = np.clip(equirect, 0.0, 1.0).astype(np.float32)
            processed_images.append(torch.from_numpy(equirect))
            pbar.update(i + 1)

        result = torch.stack(processed_images, dim=0)
        return (result,)


class EquirectangularMirrorFlip:
    """ComfyUI node to mirror/flip equirectangular images with proper spherical wrapping."""
    DESCRIPTION = "Mirror or flip an equirectangular image horizontally (longitude) or vertically (latitude) with proper spherical wrapping."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "mirror_horizontal": ("BOOLEAN", {"default": False, "tooltip": "Flip left-right: reverses longitude (horizontal 180° rotation)."}),
                "mirror_vertical": ("BOOLEAN", {"default": False, "tooltip": "Flip top-bottom: inverts latitude (vertical inversion)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("flipped_image",)
    FUNCTION = "mirror_flip"
    CATEGORY = "LatLong"

    def mirror_flip(self,
                   image: torch.Tensor,
                   mirror_horizontal: bool = False,
                   mirror_vertical: bool = False) -> Tuple[torch.Tensor]:
        batch_size = image.shape[0]
        processed_images = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_numpy = image[i].cpu().numpy()
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)

            flipped = EquirectangularProcessor.mirror_flip_equirectangular(
                img_numpy,
                mirror_horizontal=mirror_horizontal,
                mirror_vertical=mirror_vertical,
            )
            flipped = np.clip(flipped, 0.0, 1.0).astype(np.float32)
            processed_images.append(torch.from_numpy(flipped))
            pbar.update(i + 1)

        result = torch.stack(processed_images, dim=0)
        return (result,)


class EquirectangularResize:
    """ComfyUI node to resize equirectangular images with optional aspect ratio preservation."""
    DESCRIPTION = "Resize an equirectangular image with optional preservation of the standard 2:1 aspect ratio."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "output_width": ("INT", {"default": 2048, "min": 128, "max": 8192, "step": 1, "tooltip": "Target width in pixels."}),
                "output_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1, "tooltip": "Target height (ignored if maintain_aspect is enabled; auto-calculated as width/2)."}),
                "maintain_aspect": ("BOOLEAN", {"default": True, "tooltip": "Preserve 2:1 aspect ratio (standard equirectangular). Disable for custom dimensions."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling quality: lanczos (highest), bicubic, bilinear, nearest (fastest)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize"
    CATEGORY = "LatLong"

    def resize(self,
              image: torch.Tensor,
              output_width: int = 2048,
              output_height: int = 1024,
              maintain_aspect: bool = True,
              interpolation: str = "lanczos") -> Tuple[torch.Tensor]:
        batch_size = image.shape[0]
        processed_images = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_numpy = image[i].cpu().numpy()
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)

            resized = EquirectangularProcessor.resize_equirectangular(
                img_numpy,
                output_width=output_width,
                output_height=output_height,
                maintain_aspect=maintain_aspect,
                interpolation=interpolation,
            )
            resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
            processed_images.append(torch.from_numpy(resized))
            pbar.update(i + 1)

        result = torch.stack(processed_images, dim=0)
        return (result,)


class EquirectangularRotatePreset:
    """ComfyUI node to rotate equirectangular images using preset view directions."""
    DESCRIPTION = "Rotate an equirectangular image to preset views (front, back, left, right, up, down) with optional custom adjustments."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "preset": (["front", "back", "left", "right", "up", "down", "custom"], {"default": "front", "tooltip": "Quick preset views: front (0°), back (180°), left (-90°), right (90°), up/down (±90° pitch)."}),
                "yaw_offset": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Fine-tune yaw: additional horizontal rotation applied to preset (degrees)."}),
                "pitch_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1, "tooltip": "Fine-tune pitch: additional vertical tilt applied to preset (degrees)."}),
                "roll_offset": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1, "tooltip": "Fine-tune roll: additional banking rotation applied to preset (degrees)."}),
                "horizon_offset": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1, "tooltip": "Vertical horizon shift: positive moves horizon up (degrees)."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling quality: lanczos (highest), bicubic, bilinear, nearest (fastest)."}),
                "backend": (["auto", "cpu", "gpu"], {"default": "auto", "tooltip": "Processing backend: auto uses GPU if available (bilinear/nearest only on GPU)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rotated_image",)
    FUNCTION = "rotate_preset"
    CATEGORY = "LatLong"

    def rotate_preset(self,
                     image: torch.Tensor,
                     preset: str = "front",
                     yaw_offset: float = 0.0,
                     pitch_offset: float = 0.0,
                     roll_offset: float = 0.0,
                     horizon_offset: float = 0.0,
                     interpolation: str = "lanczos",
                     backend: str = "auto") -> Tuple[torch.Tensor]:
        # Get preset rotation values
        preset_yaw, preset_pitch, preset_roll = EquirectangularProcessor.get_preset_rotation(preset)

        # Apply offsets
        final_yaw = preset_yaw + yaw_offset
        final_pitch = preset_pitch + pitch_offset
        final_roll = preset_roll + roll_offset

        batch_size = image.shape[0]
        processed_images = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_tensor = image[i]
            img_numpy = img_tensor.cpu().numpy()

            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)

            # Choose backend
            use_gpu = (backend == 'gpu') or (backend == 'auto' and torch.cuda.is_available())
            if use_gpu:
                img_dev = image[i].to('cuda')
                proc_t = EquirectangularProcessor.torch_rotate_equirectangular(
                    img_dev,
                    yaw=final_yaw,
                    pitch=final_pitch,
                    roll=final_roll,
                    horizon_offset=horizon_offset,
                    interpolation=interpolation if interpolation in ('bilinear', 'nearest') else 'bilinear'
                ).to('cpu').numpy()
                processed_img = proc_t
            else:
                processed_img = EquirectangularProcessor.rotate_equirectangular(
                    img_numpy,
                    yaw=final_yaw,
                    pitch=final_pitch,
                    roll=final_roll,
                    horizon_offset=horizon_offset,
                    interpolation=interpolation
                )

            processed_img = np.clip(processed_img, 0.0, 1.0).astype(np.float32)
            processed_tensor = torch.from_numpy(processed_img)
            processed_images.append(processed_tensor)

            pbar.update(i + 1)

        result = torch.stack(processed_images, dim=0)
        return (result,)


class CubemapFacesExtract:
    """ComfyUI node to extract individual cube faces from an equirectangular image."""
    DESCRIPTION = "Extract individual cube faces from an equirectangular panorama, returned as separate images."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1]."}),
            },
            "optional": {
                "face_size": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1, "tooltip": "Resolution per cube face in pixels (each output will be face_size × face_size)."}),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "Resampling quality: lanczos (highest), bicubic, bilinear, nearest (fastest)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("left", "front", "right", "back", "top", "bottom")
    FUNCTION = "extract_faces"
    CATEGORY = "LatLong"

    def extract_faces(self,
                     image: torch.Tensor,
                     face_size: int = 512,
                     interpolation: str = "lanczos") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = image.shape[0]

        left_faces = []
        front_faces = []
        right_faces = []
        back_faces = []
        top_faces = []
        bottom_faces = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_numpy = image[i].cpu().numpy()
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)

            # Generate cubemap atlas (3x2)
            atlas = EquirectangularProcessor.equirectangular_to_cubemap(
                img_numpy,
                face_size=face_size,
                layout="3x2",
                interpolation=interpolation,
            )

            # Extract individual faces from atlas
            # Layout: [left, front, right] / [back, top, bottom]
            left = atlas[0:face_size, 0:face_size]
            front = atlas[0:face_size, face_size:2*face_size]
            right = atlas[0:face_size, 2*face_size:3*face_size]
            back = atlas[face_size:2*face_size, 0:face_size]
            top = atlas[face_size:2*face_size, face_size:2*face_size]
            bottom = atlas[face_size:2*face_size, 2*face_size:3*face_size]

            # Convert to tensors and append
            left_faces.append(torch.from_numpy(np.clip(left, 0.0, 1.0).astype(np.float32)))
            front_faces.append(torch.from_numpy(np.clip(front, 0.0, 1.0).astype(np.float32)))
            right_faces.append(torch.from_numpy(np.clip(right, 0.0, 1.0).astype(np.float32)))
            back_faces.append(torch.from_numpy(np.clip(back, 0.0, 1.0).astype(np.float32)))
            top_faces.append(torch.from_numpy(np.clip(top, 0.0, 1.0).astype(np.float32)))
            bottom_faces.append(torch.from_numpy(np.clip(bottom, 0.0, 1.0).astype(np.float32)))

            pbar.update(i + 1)

        # Stack into batches
        left_batch = torch.stack(left_faces, dim=0)
        front_batch = torch.stack(front_faces, dim=0)
        right_batch = torch.stack(right_faces, dim=0)
        back_batch = torch.stack(back_faces, dim=0)
        top_batch = torch.stack(top_faces, dim=0)
        bottom_batch = torch.stack(bottom_faces, dim=0)

        return (left_batch, front_batch, right_batch, back_batch, top_batch, bottom_batch)


class PanoramaViewerNode:
    """A ComfyUI node that provides an interactive 360-degree panorama viewer for
    equirectangular images.

    This node takes an input image tensor and displays it in a Three.js-based
    panoramic viewer that allows users to:
    - Pan around the 360-degree view using mouse drag
    - Zoom in/out using the mouse wheel
    - View the panorama with proper equirectangular projection

    The viewer automatically handles different image formats and sizes, including:
    - Batch processing (takes first image from batch)
    - Grayscale to RGB conversion
    - Image resizing for performance optimization
    - Float to uint8 conversion
    """
    DESCRIPTION = "Interactive 360° panorama viewer. Drag to look around, scroll to zoom."

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node.

        Returns:
            dict: Dictionary containing required input parameters:
                - images (torch.Tensor): Input tensor containing the panoramic image.
                - max_width (int): Maximum dimension for resizing. Default: 4096
                  Set to -1 for no resizing.
        """
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Equirectangular panoramic image to display interactively."}),
                "max_width": (
                    "INT",
                    {
                        "default": 4096,
                        "min": -1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "The max width to use. Images larger than the"
                        + " specified value will be resized. Larger sizes may run"
                        + " slower. Set to -1 for no resizing.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "view_pano"
    OUTPUT_NODE = True
    CATEGORY = "LatLong"

    def __init__(self) -> None:
        self.type = "pano"

    def view_pano(
        self, images: torch.Tensor, max_width: int = 4096
    ) -> Dict[str, Dict[str, str]]:
        """Process and display the panoramic image in the viewer.

        This method handles the conversion of the input tensor to a viewable format by:
        1. Extracting the first image if dealing with a batch
        2. Converting the tensor to a numpy array
        3. Converting float values to uint8 if necessary
        4. Converting grayscale to RGB if necessary
        5. Resizing the image if it exceeds max_width
        6. Converting the processed image to a base64-encoded PNG

        Args:
            images (torch.Tensor): Input tensor containing the panoramic image(s).
                Should be in format (B, H, W, C) or (H, W, C).
            max_width (int, optional): Maximum width for resizing. Images will not
                resized if they are smaller than the specified size. Set to -1 to
                disable resizing. Default: 4096

        Returns:
            Dict[str, Dict[str, str]]: Dictionary containing the UI update information
                with the base64-encoded PNG image data.
        """
        # Handle batch dimension
        if len(images.shape) == 4:
            image = images[0]
        else:
            image = images

        # Convert to numpy and proper format
        image_np = image.cpu().numpy()

        # Convert to uint8 if needed
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        # Handle grayscale images
        if len(image_np.shape) == 2 or image_np.shape[2] == 1:
            image_np = np.repeat(image_np[..., np.newaxis], 3, axis=2)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)

        # Optionally resize image
        if max_width > 0 and (
            pil_image.size[0] > max_width or pil_image.size[1] > max_width
        ):
            new_size = tuple(
                [int(max_width * x / max(pil_image.size)) for x in pil_image.size]
            )
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.LANCZOS)  # type: ignore[arg-type]

        # Save to BytesIO
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")

        # Get base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"ui": {"pano_image": f"data:image/png;base64,{img_str}"}}


class PanoramaVideoViewerNode:
    """A ComfyUI node that provides an interactive 360-degree panorama viewer for
    equirectangular video content.

    This node takes a video file path or a sequence of image frames and displays it in
    a Three.js-based panoramic video viewer that allows users to:
    - Pan around the 360-degree view using mouse drag
    - Zoom in/out using the mouse wheel
    - Play, pause, and scrub through the video timeline
    - View the panorama video with proper equirectangular projection

    The viewer automatically handles different video formats and sizes, including:
    - Batch processing (takes frames from batch)
    - Proper frame sequencing and timing
    - Image resizing for performance optimization
    - Float to uint8 conversion
    """
    DESCRIPTION = "Interactive 360° panorama video viewer with playback controls."

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node.

        Returns:
            dict: Dictionary containing required input parameters:
                - video_frames (torch.Tensor): Input tensor containing a batch of video frames.
                - fps (int): Frames per second for the video playback.
                - max_width (int): Maximum dimension for resizing. Default: 2048
                  Set to -1 for no resizing.
        """
        return {
            "required": {
                "video_frames": ("IMAGE", {"tooltip": "Batch of equirectangular frames for 360° video playback."}),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 120,
                        "step": 1,
                        "tooltip": "Frames per second for video playback",
                    },
                ),
                "max_width": (
                    "INT",
                    {
                        "default": 2048,
                        "min": -1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "The max width to use. Frames larger than the"
                        + " specified value will be resized. Larger sizes may run"
                        + " slower. Set to -1 for no resizing.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "view_video_pano"
    OUTPUT_NODE = True
    CATEGORY = "LatLong"

    def __init__(self) -> None:
        self.type = "video_pano"

    def view_video_pano(
        self, video_frames: torch.Tensor, fps: int = 30, max_width: int = 2048
    ) -> Dict[str, Dict[str, str]]:
        """Process and display the panoramic video in the viewer.

        This method handles the conversion of the input tensor frames to a viewable video format by:
        1. Processing each frame in the batch
        2. Converting each tensor to a numpy array
        3. Converting float values to uint8 if necessary
        4. Converting grayscale to RGB if necessary
        5. Resizing the frames if they exceed max_width
        6. Creating a base64-encoded video from the frames

        Args:
            video_frames (torch.Tensor): Input tensor containing the panoramic video frames.
                Should be in format (B, H, W, C).
            fps (int, optional): Frames per second for video playback. Default: 30
            max_width (int, optional): Maximum width for resizing. Frames will not be
                resized if they are smaller than the specified size. Set to -1 to
                disable resizing. Default: 2048

        Returns:
            Dict[str, Dict[str, str]]: Dictionary containing the UI update information
                with the base64-encoded video data and playback parameters.
        """
        # Ensure we have batch dimension
        if len(video_frames.shape) != 4:
            raise ValueError("Expected video frames in batch format (B, H, W, C)")

        # Create a list to store processed frames
        processed_frames = []

        # Process each frame
        for frame in video_frames:
            # Convert to numpy and proper format
            frame_np = frame.cpu().numpy()

            # Convert to uint8 if needed
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).astype(np.uint8)

            # Handle grayscale images
            if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:
                frame_np = np.repeat(frame_np[..., np.newaxis], 3, axis=2)

            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_np)

            # Optionally resize image
            if max_width > 0 and (
                pil_frame.size[0] > max_width or pil_frame.size[1] > max_width
            ):
                new_size = tuple(
                    [int(max_width * x / max(pil_frame.size)) for x in pil_frame.size]
                )
                pil_frame = pil_frame.resize(
                    new_size, resample=Image.Resampling.LANCZOS
                )

            # Add to processed frames
            processed_frames.append(pil_frame)

        # Check if we have any frames
        if not processed_frames:
            return {"ui": {"error": "No frames found in input"}}

        # Create a list to store base64 strings of each frame
        frame_data = []

        # Convert each frame to base64
        for frame in processed_frames:
            buffered = BytesIO()
            frame.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            frame_data.append(f"data:image/png;base64,{img_str}")

        # Return the frame data, count, and fps
        return {
            "ui": {
                "pano_video_preview": frame_data[0],  # First frame as preview
                "pano_video_frames": frame_data,  # All frames as list of strings
                "frame_count": str(len(processed_frames)),
                "fps": str(fps),
                "video_type": "360_equirectangular",
            }
        }
