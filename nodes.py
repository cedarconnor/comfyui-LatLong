import os
import copy
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import base64
from io import BytesIO

import folder_paths
from comfy.utils import ProgressBar
from PIL import Image

from .modules.equirectangular_processor import EquirectangularProcessor
import math


custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))


_CUBEMAP_CANONICAL_FACES = ("front", "right", "back", "left", "up", "down")
_CUBEMAP_FACE_SYNONYMS = {
    "f": "front",
    "front": "front",
    "r": "right",
    "right": "right",
    "b": "back",
    "back": "back",
    "l": "left",
    "left": "left",
    "u": "up",
    "up": "up",
    "top": "up",
    "d": "down",
    "down": "down",
    "bottom": "down",
}


def _tokenize_face_order(face_order: str) -> List[str]:
    normalized = face_order.replace("|", ",").replace(";", ",")
    tokens: List[str] = []
    for part in normalized.split(","):
        for token in part.strip().split():
            if token:
                tokens.append(token)
    return tokens


def _parse_face_order(face_order: str) -> List[str]:
    tokens = _tokenize_face_order(face_order)
    faces: List[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key not in _CUBEMAP_FACE_SYNONYMS:
            raise ValueError(
                f"Unknown face token '{token}'. Use one of: "
                + ", ".join(sorted(set(_CUBEMAP_FACE_SYNONYMS.keys())))
            )
        faces.append(_CUBEMAP_FACE_SYNONYMS[key])

    if len(faces) != 6:
        raise ValueError(
            f"face_order must specify 6 faces, got {len(faces)}: {faces}. "
            "Example: 'F,R,B,L,U,D'"
        )
    if len(set(faces)) != 6:
        raise ValueError(
            f"face_order must contain each face exactly once, got: {faces}"
        )
    return faces


def _faces_from_atlas_3x2(atlas: np.ndarray, face_size: int) -> Dict[str, np.ndarray]:
    return {
        "left": atlas[0:face_size, 0:face_size],
        "front": atlas[0:face_size, face_size : 2 * face_size],
        "right": atlas[0:face_size, 2 * face_size : 3 * face_size],
        "back": atlas[face_size : 2 * face_size, 0:face_size],
        "up": atlas[face_size : 2 * face_size, face_size : 2 * face_size],
        "down": atlas[face_size : 2 * face_size, 2 * face_size : 3 * face_size],
    }


def _ensure_face_3d(face: np.ndarray) -> np.ndarray:
    return face[:, :, None] if face.ndim == 2 else face


def _atlas_3x2_from_faces(faces: Dict[str, np.ndarray]) -> np.ndarray:
    face = _ensure_face_3d(faces["front"])
    face_size = int(face.shape[0])
    channels = int(face.shape[2])
    atlas = np.zeros((face_size * 2, face_size * 3, channels), dtype=face.dtype)
    atlas[0:face_size, 0:face_size] = _ensure_face_3d(faces["left"])
    atlas[0:face_size, face_size : 2 * face_size] = _ensure_face_3d(faces["front"])
    atlas[0:face_size, 2 * face_size : 3 * face_size] = _ensure_face_3d(faces["right"])
    atlas[face_size : 2 * face_size, 0:face_size] = _ensure_face_3d(faces["back"])
    atlas[face_size : 2 * face_size, face_size : 2 * face_size] = _ensure_face_3d(faces["up"])
    atlas[face_size : 2 * face_size, 2 * face_size : 3 * face_size] = _ensure_face_3d(faces["down"])
    return atlas


def _dice_from_faces(faces: Dict[str, np.ndarray]) -> np.ndarray:
    face = _ensure_face_3d(faces["front"])
    face_size = int(face.shape[0])
    channels = int(face.shape[2])
    dice = np.zeros((face_size * 3, face_size * 4, channels), dtype=face.dtype)

    # Standard dice/cross layout:
    #   [   ][ up ][   ][   ]
    #   [left][front][right][back]
    #   [   ][down][   ][   ]
    dice[0:face_size, face_size : 2 * face_size] = _ensure_face_3d(faces["up"])
    dice[face_size : 2 * face_size, 0:face_size] = _ensure_face_3d(faces["left"])
    dice[face_size : 2 * face_size, face_size : 2 * face_size] = _ensure_face_3d(faces["front"])
    dice[face_size : 2 * face_size, 2 * face_size : 3 * face_size] = _ensure_face_3d(faces["right"])
    dice[face_size : 2 * face_size, 3 * face_size : 4 * face_size] = _ensure_face_3d(faces["back"])
    dice[2 * face_size : 3 * face_size, face_size : 2 * face_size] = _ensure_face_3d(faces["down"])
    return dice


def _faces_from_dice(dice: np.ndarray, face_size: int) -> Dict[str, np.ndarray]:
    return {
        "up": dice[0:face_size, face_size : 2 * face_size],
        "left": dice[face_size : 2 * face_size, 0:face_size],
        "front": dice[face_size : 2 * face_size, face_size : 2 * face_size],
        "right": dice[face_size : 2 * face_size, 2 * face_size : 3 * face_size],
        "back": dice[face_size : 2 * face_size, 3 * face_size : 4 * face_size],
        "down": dice[2 * face_size : 3 * face_size, face_size : 2 * face_size],
    }


def _horizon_from_faces(faces: Dict[str, np.ndarray], face_order: List[str]) -> np.ndarray:
    face = _ensure_face_3d(faces["front"])
    face_size = int(face.shape[0])
    channels = int(face.shape[2])
    horizon = np.zeros((face_size, face_size * 6, channels), dtype=face.dtype)
    for idx, name in enumerate(face_order):
        horizon[:, idx * face_size : (idx + 1) * face_size] = _ensure_face_3d(faces[name])
    return horizon


def _faces_from_horizon(horizon: np.ndarray, face_order: List[str]) -> Dict[str, np.ndarray]:
    h, w = horizon.shape[:2]
    if w % 6 != 0:
        raise ValueError(f"Horizon cubemap width must be divisible by 6, got {w}")
    face_size = w // 6
    if h != face_size:
        raise ValueError(
            f"Horizon cubemap must have H==W/6 (faces are square), got H={h}, W={w}"
        )

    faces: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(face_order):
        faces[name] = horizon[:, idx * face_size : (idx + 1) * face_size]
    return faces


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
                "use_tiling": (["auto", "enabled", "disabled"], {"default": "auto", "tooltip": "Tiled processing for 16K+ images. Auto enables for >8K images. Reduces memory usage."}),
                "tile_size": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 512, "tooltip": "Tile size in pixels for tiled processing. Smaller = less memory, more tiles."}),
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
                             backend: str = "auto",
                             use_tiling: str = "auto",
                             tile_size: int = 2048) -> Tuple[torch.Tensor]:
        
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

            # Determine tiling setting
            use_tiling_bool = None  # Let processor auto-decide
            if use_tiling == "enabled":
                use_tiling_bool = True
            elif use_tiling == "disabled":
                use_tiling_bool = False
            # else: "auto" means None, which auto-decides based on image size

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
                    interpolation=interpolation,
                    use_tiling=use_tiling_bool,
                    tile_size=tile_size
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


class EquirectangularToCubemapFlexible:
    """ComfyUI node to convert an equirectangular image into multiple cubemap formats."""

    DESCRIPTION = "Convert equirectangular panoramas into cubemaps in multiple formats (3x2 atlas, dice, horizon, stack, list, dict)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Equirectangular input image tensor (B,H,W,C) in [0,1].",
                    },
                ),
            },
            "optional": {
                "face_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Resolution (pixels) per cube face.",
                    },
                ),
                "cube_format": (
                    ["atlas_3x2", "dice", "horizon", "stack", "list", "dict"],
                    {
                        "default": "atlas_3x2",
                        "tooltip": "Output cubemap layout: 3x2 atlas, dice cross, horizon strip, or stack (B*6 faces). Note: list/dict are aliases of stack for ComfyUI compatibility.",
                    },
                ),
                "face_order": (
                    "STRING",
                    {
                        "default": "F,R,B,L,U,D",
                        "multiline": False,
                        "tooltip": "Face order for stack/horizon (and list/dict alias) formats. Tokens: F,R,B,L,U,D (also accepts front/right/back/left/up/down). Example: 'F,R,B,L,U,D'.",
                    },
                ),
                "interpolation": (
                    ["lanczos", "bicubic", "bilinear", "nearest"],
                    {
                        "default": "lanczos",
                        "tooltip": "Resampling filter used during face generation.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cubemap",)
    FUNCTION = "to_cubemap_flexible"
    CATEGORY = "LatLong/Cubemap"

    def to_cubemap_flexible(
        self,
        image: torch.Tensor,
        face_size: int = 512,
        cube_format: str = "atlas_3x2",
        face_order: str = "F,R,B,L,U,D",
        interpolation: str = "lanczos",
    ) -> Tuple[Any]:
        batch_size = int(image.shape[0])
        pbar = ProgressBar(batch_size)

        order: Optional[List[str]] = None
        if cube_format in ("horizon", "stack", "list", "dict"):
            order = _parse_face_order(face_order)

        processed_images: List[np.ndarray] = []
        stacked_faces: List[np.ndarray] = []

        for i in range(batch_size):
            img_numpy = image[i].cpu().numpy()
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)

            atlas = EquirectangularProcessor.equirectangular_to_cubemap(
                img_numpy, face_size=face_size, layout="3x2", interpolation=interpolation
            )
            atlas = np.clip(atlas, 0.0, 1.0).astype(np.float32)
            faces = _faces_from_atlas_3x2(atlas, face_size)

            if cube_format == "atlas_3x2":
                processed_images.append(atlas)
            elif cube_format == "dice":
                processed_images.append(_dice_from_faces(faces))
            elif cube_format == "horizon":
                assert order is not None
                processed_images.append(_horizon_from_faces(faces, order))
            elif cube_format in ("stack", "list", "dict"):
                assert order is not None
                for name in order:
                    stacked_faces.append(faces[name])
            else:
                raise ValueError(f"Unknown cube_format: {cube_format}")

            pbar.update(i + 1)

        if cube_format in ("stack", "list", "dict"):
            stacked = np.stack(stacked_faces, axis=0).astype(np.float32)
            return (torch.from_numpy(stacked),)

        if cube_format in ("atlas_3x2", "dice", "horizon"):
            result = torch.from_numpy(np.stack(processed_images, axis=0).astype(np.float32))
            return (result,)

        raise ValueError(f"Unsupported cube_format: {cube_format}")


class CubemapToEquirectangularFlexible:
    """ComfyUI node to convert multiple cubemap formats back to equirectangular."""

    DESCRIPTION = "Convert cubemaps in multiple formats (3x2 atlas, dice, horizon, stack, list, dict) back to equirectangular."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap": (
                    "IMAGE",
                    {
                        "tooltip": "Cubemap input tensor. Use cube_format=atlas_3x2/dice/horizon for packed layouts, or stack/list/dict (alias) for face stacks (B*6 faces).",
                    },
                ),
            },
            "optional": {
                "cube_format": (
                    ["atlas_3x2", "dice", "horizon", "stack", "list", "dict"],
                    {
                        "default": "atlas_3x2",
                        "tooltip": "How to interpret the input cubemap. Note: list/dict are aliases of stack for ComfyUI compatibility.",
                    },
                ),
                "face_order": (
                    "STRING",
                    {
                        "default": "F,R,B,L,U,D",
                        "multiline": False,
                        "tooltip": "Face order for stack/horizon (and list/dict alias) formats. Must match how the cubemap was packed.",
                    },
                ),
                "output_width": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Width of output equirectangular panorama (typically 2x height for standard 2:1 ratio).",
                    },
                ),
                "output_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 128,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Height of output equirectangular panorama.",
                    },
                ),
                "interpolation": (
                    ["lanczos", "bicubic", "bilinear", "nearest"],
                    {
                        "default": "lanczos",
                        "tooltip": "Resampling quality when mapping cubemap -> equirectangular.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("equirectangular_image",)
    FUNCTION = "to_equirectangular_flexible"
    CATEGORY = "LatLong/Cubemap"

    def to_equirectangular_flexible(
        self,
        cubemap: Any,
        cube_format: str = "atlas_3x2",
        face_order: str = "F,R,B,L,U,D",
        output_width: int = 2048,
        output_height: int = 1024,
        interpolation: str = "lanczos",
    ) -> Tuple[torch.Tensor]:
        order: Optional[List[str]] = None
        if cube_format in ("horizon", "stack", "list", "dict"):
            order = _parse_face_order(face_order)

        processed_images: List[torch.Tensor] = []

        if not isinstance(cubemap, torch.Tensor):
            raise ValueError(f"Unsupported cubemap input type for cube_format={cube_format}: {type(cubemap)}")

        if cube_format in ("stack", "list", "dict"):
            assert order is not None
            total_faces = int(cubemap.shape[0])
            if total_faces % 6 != 0:
                raise ValueError(f"Stack cubemap expects N faces divisible by 6, got {total_faces}")
            num_panos = total_faces // 6
            pbar = ProgressBar(num_panos)

            for pano_idx in range(num_panos):
                group = cubemap[pano_idx * 6 : (pano_idx + 1) * 6]  # (6,H,W,C)
                if group.shape[1] != group.shape[2]:
                    raise ValueError("Stack cubemap faces must be square (H==W)")

                faces: Dict[str, np.ndarray] = {}
                for idx, name in enumerate(order):
                    face_np = group[idx].cpu().numpy()
                    if face_np.dtype != np.float32:
                        face_np = face_np.astype(np.float32)
                    faces[name] = face_np

                atlas = _atlas_3x2_from_faces(faces)
                equi = EquirectangularProcessor.cubemap_to_equirectangular(
                    atlas,
                    output_width=output_width,
                    output_height=output_height,
                    layout="3x2",
                    interpolation=interpolation,
                )
                equi = np.clip(equi, 0.0, 1.0).astype(np.float32)
                processed_images.append(torch.from_numpy(equi))
                pbar.update(pano_idx + 1)

            return (torch.stack(processed_images, dim=0),)

        batch_size = int(cubemap.shape[0])
        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            cube_np = cubemap[i].cpu().numpy()
            if cube_np.dtype != np.float32:
                cube_np = cube_np.astype(np.float32)

            if cube_format == "atlas_3x2":
                atlas = cube_np
            elif cube_format == "dice":
                face_size = cube_np.shape[0] // 3
                faces = _faces_from_dice(cube_np, face_size)
                atlas = _atlas_3x2_from_faces(faces)
            elif cube_format == "horizon":
                assert order is not None
                faces = _faces_from_horizon(cube_np, order)
                atlas = _atlas_3x2_from_faces(faces)
            else:
                raise ValueError(f"Unknown cube_format: {cube_format}")

            equi = EquirectangularProcessor.cubemap_to_equirectangular(
                atlas,
                output_width=output_width,
                output_height=output_height,
                layout="3x2",
                interpolation=interpolation,
            )
            equi = np.clip(equi, 0.0, 1.0).astype(np.float32)
            processed_images.append(torch.from_numpy(equi))
            pbar.update(i + 1)

        return (torch.stack(processed_images, dim=0),)


class StackCubemapFacesNode:
    """Stack six cubemap faces into a single face-stack tensor (B*6, H, W, C)."""

    DESCRIPTION = "Stack cubemap faces into a single tensor (B*6 faces). Use with cube_format='stack'."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Front": ("IMAGE", {"tooltip": "Front face image batch."}),
                "Right": ("IMAGE", {"tooltip": "Right face image batch."}),
                "Back": ("IMAGE", {"tooltip": "Back face image batch."}),
                "Left": ("IMAGE", {"tooltip": "Left face image batch."}),
                "Up": ("IMAGE", {"tooltip": "Up face image batch."}),
                "Down": ("IMAGE", {"tooltip": "Down face image batch."}),
            },
            "optional": {
                "face_order": (
                    "STRING",
                    {
                        "default": "F,R,B,L,U,D",
                        "multiline": False,
                        "tooltip": "Output stack face order. Must match the order you use when converting back from stack.",
                    },
                )
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_stack",)
    FUNCTION = "stack_faces"
    CATEGORY = "LatLong/Cubemap"

    def stack_faces(
        self,
        Front: torch.Tensor,
        Right: torch.Tensor,
        Back: torch.Tensor,
        Left: torch.Tensor,
        Up: torch.Tensor,
        Down: torch.Tensor,
        face_order: str = "F,R,B,L,U,D",
    ) -> Tuple[torch.Tensor]:
        order = _parse_face_order(face_order)
        faces = {
            "front": Front,
            "right": Right,
            "back": Back,
            "left": Left,
            "up": Up,
            "down": Down,
        }

        # Validate shapes
        shapes = {name: tuple(t.shape) for name, t in faces.items()}
        batch_sizes = {shape[0] for shape in shapes.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"All faces must have same batch size, got: {shapes}")

        b, h, w, c = Front.shape
        for name, t in faces.items():
            if tuple(t.shape) != (b, h, w, c):
                raise ValueError(f"All faces must match shape (B,H,W,C); {name} got {tuple(t.shape)} vs {(b,h,w,c)}")

        stacked: List[torch.Tensor] = []
        for i in range(b):
            for name in order:
                stacked.append(faces[name][i])

        return (torch.stack(stacked, dim=0),)


class SplitCubemapFacesNode:
    """Split a face-stack tensor (B*6, H, W, C) into six face batches (B, H, W, C)."""

    DESCRIPTION = "Split a cubemap face-stack into six face batches. Supports any stack face order."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_stack": ("IMAGE", {"tooltip": "Stacked cubemap faces (B*6, H, W, C)."}),
            },
            "optional": {
                "face_order": (
                    "STRING",
                    {
                        "default": "F,R,B,L,U,D",
                        "multiline": False,
                        "tooltip": "Input stack face order. Must match how the stack was created.",
                    },
                )
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Front", "Right", "Back", "Left", "Up", "Down")
    FUNCTION = "split_faces"
    CATEGORY = "LatLong/Cubemap"

    def split_faces(
        self, face_stack: torch.Tensor, face_order: str = "F,R,B,L,U,D"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        order = _parse_face_order(face_order)
        total_faces = int(face_stack.shape[0])
        if total_faces % 6 != 0:
            raise ValueError(f"face_stack first dim must be divisible by 6, got {total_faces}")
        batch_size = total_faces // 6

        # For each panorama, pick the face at the correct position in the stack.
        def gather_face(face_name: str) -> torch.Tensor:
            idx_in_stack = order.index(face_name)
            gathered = [face_stack[i * 6 + idx_in_stack] for i in range(batch_size)]
            return torch.stack(gathered, dim=0)

        front = gather_face("front")
        right = gather_face("right")
        back = gather_face("back")
        left = gather_face("left")
        up = gather_face("up")
        down = gather_face("down")
        return (front, right, back, left, up, down)


def _create_center_seam_mask(
    x: torch.Tensor, frac_width: float = 0.10, pixel_width: int = 0, feather: int = 0
) -> torch.Tensor:
    """Create a vertical seam mask centered in the image (BHWC input -> BHW mask)."""
    b, h, w, *_ = x.shape

    if pixel_width > 0:
        strip = int(pixel_width)
    else:
        strip = max(1, int(w * float(frac_width)))

    strip = max(1, min(strip, w))
    x0 = (w - strip) // 2
    x1 = x0 + strip

    mask = torch.zeros((b, h, w), dtype=x.dtype, device=x.device)
    if feather <= 0:
        mask[:, :, x0:x1] = 1.0
        return mask

    feather = int(feather)

    # Left feather
    left_start = max(0, x0 - feather)
    left_end = x0
    if left_end > left_start:
        steps = torch.linspace(
            0.0,
            1.0,
            left_end - left_start,
            dtype=x.dtype,
            device=x.device,
        )
        mask[:, :, left_start:left_end] = steps[None, None, :]

    # Center strip
    mask[:, :, x0:x1] = 1.0

    # Right feather
    right_start = x1
    right_end = min(w, x1 + feather)
    if right_end > right_start:
        steps = torch.linspace(
            1.0,
            0.0,
            right_end - right_start,
            dtype=x.dtype,
            device=x.device,
        )
        mask[:, :, right_start:right_end] = steps[None, None, :]

    return mask


def _circle_mask_np(
    size: int, circle_radius: float, pixel_radius: int = 0, feather: int = 0
) -> np.ndarray:
    """Create a centered circular mask in a square array, values in [0,1]."""
    size = int(size)
    if size < 1:
        raise ValueError("size must be >= 1")

    max_radius = size / 2.0
    inner_radius = float(pixel_radius) if pixel_radius > 0 else float(circle_radius) * max_radius
    feather = int(feather)
    outer_radius = inner_radius + float(max(0, feather))

    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)

    mask = np.zeros((size, size), dtype=np.float32)
    mask[dist <= inner_radius] = 1.0

    if feather > 0:
        zone = (dist > inner_radius) & (dist <= outer_radius)
        mask[zone] = 1.0 - (dist[zone] - inner_radius) / float(feather)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


class LatLongCreateSeamMask:
    """Create a seam mask for inpainting equirectangular seams."""

    DESCRIPTION = "Create a centered vertical seam mask (optionally feathered). Useful with Roll Image for seam inpainting."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Reference image (B,H,W,C) used to size the mask."}),
            },
            "optional": {
                "frac_width": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Fraction of image width for the seam strip (ignored if pixel_width > 0).",
                    },
                ),
                "pixel_width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Explicit seam width in pixels (overrides frac_width if > 0).",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 1,
                        "tooltip": "Feather width in pixels on both sides of the seam.",
                    },
                ),
                "roll_x_by_50_percent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Shift the mask horizontally by 50% (useful if you rolled the image by 180°).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("seam_mask",)
    FUNCTION = "run"
    CATEGORY = "LatLong/Mask"

    def run(
        self,
        image: torch.Tensor,
        frac_width: float = 0.10,
        pixel_width: int = 0,
        feather: int = 0,
        roll_x_by_50_percent: bool = False,
    ) -> Tuple[torch.Tensor]:
        if image.dim() != 4:
            raise ValueError(f"Expected IMAGE (B,H,W,C), got shape {tuple(image.shape)}")
        mask = _create_center_seam_mask(
            image, frac_width=frac_width, pixel_width=pixel_width, feather=feather
        )
        if roll_x_by_50_percent:
            shift = int(image.shape[2]) // 2
            mask = torch.roll(mask, shifts=(0, shift), dims=(1, 2))
        return (mask,)


class LatLongCreatePoleMask:
    """Create a pole mask for inpainting poles on cubemap faces or equirectangular panoramas."""

    DESCRIPTION = "Create a circular pole mask. Mode 'face' outputs a centered circle mask on a face. Mode 'equirectangular' maps circles on Up/Down cube faces back to equirectangular."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Reference IMAGE (face or equirectangular) used to size the mask."}),
                "circle_radius": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Circle radius as a fraction of the max possible radius (min(H,W)/2). Ignored if pixel_radius > 0.",
                    },
                ),
                "pixel_radius": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Circle radius in pixels (overrides circle_radius if > 0).",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Feather width in pixels.",
                    },
                ),
                "mode": (
                    ["face", "equirectangular"],
                    {
                        "default": "face",
                        "tooltip": "face: mask matches input face dimensions. equirectangular: creates masks on Up/Down cube faces then maps back to equirectangular.",
                    },
                ),
            },
            "optional": {
                "face_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": -1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Only used for mode=equirectangular. Cubemap face size to generate the pole mask. Set to -1 for auto (H//2).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("pole_mask",)
    FUNCTION = "run"
    CATEGORY = "LatLong/Mask"

    def run(
        self,
        image: torch.Tensor,
        circle_radius: float = 0.10,
        pixel_radius: int = 0,
        feather: int = 0,
        mode: str = "face",
        face_size: int = 512,
    ) -> Tuple[torch.Tensor]:
        if image.dim() != 4:
            raise ValueError(f"Expected IMAGE (B,H,W,C), got shape {tuple(image.shape)}")

        b, h, w, _ = image.shape

        if mode == "face":
            if h != w:
                raise ValueError(f"face mode expects square faces (H==W), got H={h}, W={w}")
            mask_2d = _circle_mask_np(
                size=h, circle_radius=circle_radius, pixel_radius=pixel_radius, feather=feather
            )
            mask = torch.from_numpy(mask_2d).to(dtype=image.dtype, device=image.device)
            return (mask.unsqueeze(0).repeat(b, 1, 1),)

        # mode == "equirectangular"
        fs = int(face_size) if int(face_size) > 0 else max(16, int(h) // 2)
        circle = _circle_mask_np(
            size=fs, circle_radius=circle_radius, pixel_radius=pixel_radius, feather=feather
        )
        faces = {
            "front": np.zeros((fs, fs), dtype=np.float32),
            "right": np.zeros((fs, fs), dtype=np.float32),
            "back": np.zeros((fs, fs), dtype=np.float32),
            "left": np.zeros((fs, fs), dtype=np.float32),
            "up": circle,
            "down": circle,
        }
        atlas = _atlas_3x2_from_faces(faces).astype(np.float32)
        pole_mask = EquirectangularProcessor.cubemap_to_equirectangular(
            atlas,
            output_width=int(w),
            output_height=int(h),
            layout="3x2",
            interpolation="bilinear",
        )
        pole_mask = np.clip(pole_mask[..., 0], 0.0, 1.0).astype(np.float32)
        mask_t = torch.from_numpy(pole_mask).to(dtype=image.dtype)
        return (mask_t.unsqueeze(0).repeat(b, 1, 1),)


class LatLongRollImage:
    """Roll an IMAGE tensor along x/y axes without resampling."""

    DESCRIPTION = "Roll an image along width/height (wraparound). Useful for moving the seam to the center before inpainting."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE", {"tooltip": "Image batch (B,H,W,C) to roll."})},
            "optional": {
                "roll_x": (
                    "INT",
                    {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Horizontal roll (pixels)."},
                ),
                "roll_y": (
                    "INT",
                    {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Vertical roll (pixels)."},
                ),
                "roll_x_by_50_percent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Set roll_x to half the image width (180° equirectangular shift). Overrides roll_x/roll_y.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rolled_image",)
    FUNCTION = "roll"
    CATEGORY = "LatLong/Utils"

    def roll(
        self, image: torch.Tensor, roll_x: int = 0, roll_y: int = 0, roll_x_by_50_percent: bool = False
    ) -> Tuple[torch.Tensor]:
        if roll_x_by_50_percent:
            roll_x = int(image.shape[2]) // 2
            roll_y = 0
        return (torch.roll(image, shifts=(int(roll_y), int(roll_x)), dims=(1, 2)),)


class LatLongRollMask:
    """Roll a MASK tensor along x/y axes."""

    DESCRIPTION = "Roll a mask along width/height (wraparound). Useful with seam masks and roll workflows."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mask": ("MASK", {"tooltip": "Mask batch (B,H,W) to roll."})},
            "optional": {
                "roll_x": (
                    "INT",
                    {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Horizontal roll (pixels)."},
                ),
                "roll_y": (
                    "INT",
                    {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Vertical roll (pixels)."},
                ),
                "roll_x_by_50_percent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Set roll_x to half the mask width. Overrides roll_x/roll_y.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("rolled_mask",)
    FUNCTION = "roll"
    CATEGORY = "LatLong/Utils"

    def roll(
        self, mask: torch.Tensor, roll_x: int = 0, roll_y: int = 0, roll_x_by_50_percent: bool = False
    ) -> Tuple[torch.Tensor]:
        if mask.dim() != 3:
            raise ValueError(f"Expected MASK (B,H,W), got shape {tuple(mask.shape)}")
        if roll_x_by_50_percent:
            roll_x = int(mask.shape[2]) // 2
            roll_y = 0
        return (torch.roll(mask, shifts=(int(roll_y), int(roll_x)), dims=(1, 2)),)


def _conv_forward_circular_x(
    self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    x = torch.nn.functional.pad(x, self.padding_values_x, mode="circular")
    x = torch.nn.functional.pad(x, self.padding_values_y, mode="constant")
    return torch.nn.functional.conv2d(
        x, weight, bias, self.stride, (0, 0), self.dilation, self.groups
    )


def _apply_circular_conv2d_padding(
    model: torch.nn.Module, is_vae: bool = False, x_axis_only: bool = True
) -> torch.nn.Module:
    modules = model.first_stage_model.modules() if is_vae else model.modules()
    for layer in modules:
        if isinstance(layer, torch.nn.Conv2d):
            if x_axis_only:
                layer.padding_values_x = (
                    layer._reversed_padding_repeated_twice[0],
                    layer._reversed_padding_repeated_twice[1],
                    0,
                    0,
                )
                layer.padding_values_y = (
                    0,
                    0,
                    layer._reversed_padding_repeated_twice[2],
                    layer._reversed_padding_repeated_twice[3],
                )
                layer._conv_forward = _conv_forward_circular_x.__get__(layer, torch.nn.Conv2d)
            else:
                layer.padding_mode = "circular"
    return model


class LatLongApplyCircularConvPaddingModel:
    """Apply circular padding to Conv2d layers in a ComfyUI MODEL to reduce seam artifacts."""

    DESCRIPTION = "Apply circular padding to MODEL Conv2d layers (x-axis only by default) for seam-safe generation."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "Model to add circular x-axis Conv2d padding to."},
                ),
                "inplace": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Modify the loaded model (True) or a copy (False). If True, reload model to restore original padding.",
                    },
                ),
                "x_axis_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply circular padding only on x-axis (recommended for equirectangular seams) or on both axes.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "LatLong/Models"

    def run(
        self, model: torch.nn.Module, inplace: bool = True, x_axis_only: bool = True
    ) -> Tuple[torch.nn.Module]:
        use_model = model if inplace else copy.deepcopy(model)
        _apply_circular_conv2d_padding(use_model.model, is_vae=False, x_axis_only=x_axis_only)
        return (use_model,)


class LatLongApplyCircularConvPaddingVAE:
    """Apply circular padding to Conv2d layers in a ComfyUI VAE to reduce seam artifacts."""

    DESCRIPTION = "Apply circular padding to VAE Conv2d layers (x-axis only by default) for seam-safe generation."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (
                    "VAE",
                    {"tooltip": "VAE to add circular x-axis Conv2d padding to."},
                ),
                "inplace": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Modify the loaded VAE (True) or a copy (False). If True, reload VAE to restore original padding.",
                    },
                ),
                "x_axis_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply circular padding only on x-axis (recommended) or on both axes.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "run"
    CATEGORY = "LatLong/Models"

    def run(
        self, vae: torch.nn.Module, inplace: bool = True, x_axis_only: bool = True
    ) -> Tuple[torch.nn.Module]:
        use_vae = vae if inplace else copy.deepcopy(vae)
        _apply_circular_conv2d_padding(use_vae, is_vae=True, x_axis_only=x_axis_only)
        return (use_vae,)


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


class EquirectangularEdgeBlender:
    """ComfyUI node for blending equirectangular edges for seamless wraparound.

    Post-processing edge blending ensures perfect wraparound continuity by smoothly
    blending the left and right edges of panoramic images. This is essential for
    preventing visible seams in 360° viewers.
    """
    DESCRIPTION = "Blend left/right edges for seamless 360° wraparound. Eliminates visible seams in panorama viewers."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular panoramic image to blend edges."}),
            },
            "optional": {
                "blend_width": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Width of blend region in pixels. 10-20 recommended for most cases. 0 disables blending."
                }),
                "blend_mode": (["cosine", "linear", "smooth"], {
                    "default": "cosine",
                    "tooltip": "Blending curve type. Cosine provides smoothest transition."
                }),
                "check_continuity": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Validate edge continuity after blending and report status."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_edges"
    CATEGORY = "LatLong"

    def blend_edges(self,
                   image: torch.Tensor,
                   blend_width: int = 10,
                   blend_mode: str = "cosine",
                   check_continuity: bool = True) -> Tuple[torch.Tensor]:

        if blend_width == 0:
            print("⚠️ blend_width=0, skipping edge blending")
            return (image,)

        batch_size = image.shape[0]
        processed_images = []

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            img_tensor = image[i]
            img_numpy = img_tensor.cpu().numpy()

            # Keep float32 for processing
            if img_numpy.dtype != np.float32:
                img_numpy = img_numpy.astype(np.float32)

            # Apply edge blending
            blended_img = EquirectangularProcessor.blend_edges(
                img_numpy,
                blend_width=blend_width,
                mode=blend_mode
            )

            # Validate seamlessness if requested
            if check_continuity:
                is_seamless = EquirectangularProcessor.check_edge_continuity(
                    blended_img,
                    threshold=0.05
                )

                if is_seamless:
                    print(f"✅ Edges blended seamlessly (mode: {blend_mode}, width: {blend_width}px)")
                else:
                    print(f"⚠️ Edges may have visible seam - try increasing blend_width or use cosine mode")

            # Ensure output is float32 in [0,1] range
            blended_img = np.clip(blended_img, 0.0, 1.0).astype(np.float32)

            blended_tensor = torch.from_numpy(blended_img)
            processed_images.append(blended_tensor)

            pbar.update(i + 1)

        result = torch.stack(processed_images, dim=0)
        return (result,)

# ============================================================================
# Outpaint Nodes
# ============================================================================

def _create_inner_feather_mask(width: int, height: int, feather: int) -> torch.Tensor:
    """Create a rectangular mask (H,W) with an inner feather gradient."""
    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    
    # Distance from nearest edge
    dist_x = torch.min(x, width - 1 - x)
    dist_y = torch.min(y, height - 1 - y)
    
    # Feather falloff
    mask_x = torch.clamp(dist_x / max(1, feather), 0, 1) if feather > 0 else (dist_x >= 0).float()
    mask_y = torch.clamp(dist_y / max(1, feather), 0, 1) if feather > 0 else (dist_y >= 0).float()
    
    # Combine (H, W)
    mask = mask_y.unsqueeze(1) * mask_x.unsqueeze(0)
    return mask

def _perspective_insert(
    flat_image: torch.Tensor,
    canvas_h: int, canvas_w: int,
    yaw: float, pitch: float, roll: float,
    fov_deg: float,
    feather: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project a flat image onto an equirectangular canvas.
    Inverse of perspective_extract.
    
    Args:
        flat_image: (C, H_src, W_src)
        canvas_h, canvas_w: Output dimensions
        yaw, pitch, roll: Camera rotation (degrees)
        fov_deg: Horizontal Field of View of the flat image
        feather: Inner feather pixels (relative to flat image source coords)
        
    Returns:
        (canvas_image (C,H,W), mask (1,H,W))
    """
    device = flat_image.device
    C, H_src, W_src = flat_image.shape
    
    # 1. Create grid for Canvas (Equirectangular)
    ys = torch.linspace(0, canvas_h - 1, canvas_h, device=device)
    xs = torch.linspace(0, canvas_w - 1, canvas_w, device=device)
    yg, xg = torch.meshgrid(ys, xs, indexing='ij') # (H, W)
    
    # Equirect -> Spherical (lat, lon)
    lon = (xg / canvas_w) * (2 * math.pi) - math.pi
    lat = (math.pi / 2) - (yg / canvas_h) * math.pi
    
    # Spherical -> Cartesian (world direction vectors)
    # Convention: +X = lon 0 (front), +Y = lon 90 (right), +Z = lat 90 (up/north pole)
    cos_lat = torch.cos(lat)
    x_world = cos_lat * torch.cos(lon)
    y_world = cos_lat * torch.sin(lon)
    z_world = torch.sin(lat)
    
    # Stack to (H, W, 3)
    xyz_world = torch.stack([x_world, y_world, z_world], dim=-1)
    
    # 2. Rotate World -> Camera (Inverse Rotation)
    # First apply user rotation, then we'll reinterpret axes so camera looks at equator by default.
    # 
    # The user's yaw/pitch should match intuitive behavior:
    # - yaw=0, pitch=0: look at center/equator (world +X)
    # - yaw rotates horizontally, pitch rotates vertically
    #
    # To achieve this, we need the camera's +Z (look direction) to map to world +X at identity.
    # Standard camera: looks down +Z with +X right, +Y down.
    # We want: camera +Z -> world +X, camera +X -> world +Y, camera +Y -> world -Z
    #
    # After applying user rotation R to world points, we remap axes:
    # cam_x = world_y, cam_y = -world_z, cam_z = world_x
    
    R = EquirectangularProcessor._torch_rotation_matrix(yaw, pitch, roll, device, torch.float32)
    # xyz_cam = xyz_world @ R (since R maps Cam->World, R.T maps World->Cam)
    xyz_rotated = torch.tensordot(xyz_world, R, dims=1)  # (H, W, 3)
    
    # Remap axes so camera looks at equator (+X) by default
    # cam_z = world_x (look direction), cam_x = world_y (right), cam_y = -world_z (down)
    x_c = xyz_rotated[..., 1]   # world_y -> cam_x
    y_c = -xyz_rotated[..., 2]  # -world_z -> cam_y (down direction)
    z_c = xyz_rotated[..., 0]   # world_x -> cam_z (look direction)
    
    # 3. Project Camera -> Flat Image Plane
    # Canonical camera looks usually down +Z or +X. 
    # In processor.perspective_extract:
    # x_cam = (u - cx) / f
    # y_cam = (v - cy) / f
    # z_cam = 1
    # This implies camera looks down +Z.
    
    # We need to filter points behind the camera
    valid_mask = z_c > 0
    
    # Perspective projection
    # u_norm = x_c / z_c, v_norm = y_c / z_c
    # y_c already accounts for down direction from axis remapping
    z_c = torch.clamp(z_c, min=1e-6)
    u_norm = x_c / z_c
    v_norm = y_c / z_c
    
    # 4. Convert to Pixel Coords
    # f = W / (2 * tan(fov/2))
    # We can normalize coordinates to [-1, 1] range first to use grid_sample
    
    # f_norm (relative to width/2) = 1.0 / tan(fov/2)
    tan_half_fov = math.tan(math.radians(fov_deg) / 2.0)
    
    # For canonical grid [-1, 1]:
    # x = u_norm * (f contribution). 
    # Let's map directly to grid coords [-1, 1] for grid_sample.
    # Horizontal: -1 is left edge, +1 is right edge of flat image.
    # Ray at edge: x/z = tan(fov/2)
    # So grid_x = (x/z) / tan(fov/2)
    grid_x = u_norm / tan_half_fov
    
    # Vertical: assume square pixels, so scale by aspect ratio
    aspect = W_src / H_src
    # grid_y = (y/z) / (tan(fov/2) / aspect)
    grid_y = v_norm / (tan_half_fov / aspect)
    
    # Update valid mask to only include points within the flat image
    valid_mask = valid_mask & (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    
    # 5. Sample from Flat Image using grid_sample
    # grid needs to be (N, H_out, W_out, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) # (1, H, W, 2)
    
    # Input image needs to be (N, C, H, W)
    img_batch = flat_image.unsqueeze(0)
    
    sampled = torch.nn.functional.grid_sample(img_batch, grid, align_corners=True, padding_mode="zeros")
    sampled = sampled.squeeze(0) # (C, H, W)
    
    # 6. Apply Feathering
    # We can compute feather in grid space
    if feather > 0:
        # Check distance from edges in pixel space approx
        # Convert grid [-1, 1] to [0, W]
        # dist_left = (grid_x - (-1)) -> ranges 0..2
        # pixel_dist = (grid_x + 1)/2 * W
        
        # Simpler: feather in normalized device coordinates (NDC)
        # NDC width = 2.0. Pixel width = W.
        # feather_ndc_x = feather / W * 2.0
        feather_ndc_x = (feather / W_src) * 2.0
        feather_ndc_y = (feather / H_src) * 2.0
        
        # Dist from nearest edge in NDC
        dist_x = 1.0 - torch.abs(grid_x)
        dist_y = 1.0 - torch.abs(grid_y)
        
        alpha_x = torch.clamp(dist_x / max(1e-6, feather_ndc_x), 0, 1)
        alpha_y = torch.clamp(dist_y / max(1e-6, feather_ndc_y), 0, 1)
        
        feather_mask = alpha_x * alpha_y
        valid_mask = valid_mask & (feather_mask > 0)
        # We'll multiply the mask later
    else:
        feather_mask = torch.ones_like(grid_x)

    # 7. Compose Final Mask
    # valid_mask is boolean (H, W).
    final_mask = valid_mask.float() * feather_mask
    
    # Zero out invalid areas in sampled image
    sampled = sampled * valid_mask.float().unsqueeze(0)
    
    return sampled, final_mask.unsqueeze(0) # (C,H,W), (1,H,W)


class LatLongOutpaintSetup:
    """Setup node for LatLong outpainting workflow."""
    
    DESCRIPTION = "Place a flat image onto a transparent equirectangular canvas for outpainting."
    CATEGORY = "LatLong/Outpaint"
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH_CONTEXT")
    RETURN_NAMES = ("composite_image", "outpaint_mask", "stitch_context")
    FUNCTION = "setup"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flat_image": ("IMAGE", {"tooltip": "Source rectilinear image to place."}),
                "canvas_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 8, "tooltip": "Width of the equirectangular canvas."}),
                "canvas_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8, "tooltip": "Height of the equirectangular canvas."}),
                "placement_mode": (["2d_composite", "perspective"], {"default": "2d_composite"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Image size multiplier. 1.0 = original size relative to canvas."}),
                "translate_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1, "tooltip": "2D only: Horizontal pixel offset."}),
                "translate_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1, "tooltip": "2D only: Vertical pixel offset."}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "tooltip": "Perspective only: Horizontal view angle."}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0, "tooltip": "Perspective only: Vertical view angle."}),
                "fov": ("FLOAT", {"default": 90.0, "min": 10.0, "max": 170.0, "step": 1.0, "tooltip": "Perspective only: Field of View at scale=1.0."}),
                "feather_size": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1, "tooltip": "Inner feathering in pixels."}),
            }
        }

    def setup(self, flat_image, canvas_width, canvas_height, placement_mode, scale, translate_x, translate_y, yaw, pitch, fov, feather_size):
        # Force 2:1 aspect ratio if requested or just use inputs? User said "force 2x1".
        # We will force height = width // 2
        canvas_height = canvas_width // 2
        
        N, H, W, C = flat_image.shape
        device = flat_image.device
        
        # Prepare batch outputs
        out_images = []
        out_masks = []
        
        # Store context (assuming N=1 for context simplicity, or list of contexts)
        # We'll store standard python objects
        context = {
            "placement_mode": placement_mode,
            "original_image": flat_image, # Keep tensor on device
            "scale": scale,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "yaw": yaw,
            "pitch": pitch,
            "fov": fov,
            "feather_size": feather_size,
            "setup_canvas_width": canvas_width,
            "setup_canvas_height": canvas_height
        }
        
        for i in range(N):
            img = flat_image[i].permute(2, 0, 1) # (C, H, W)
            
            if placement_mode == "2d_composite":
                # Create empty canvas
                canvas = torch.zeros((C, canvas_height, canvas_width), device=device, dtype=torch.float32)
                alpha = torch.zeros((1, canvas_height, canvas_width), device=device, dtype=torch.float32)
                
                # Calculate scaled dimensions
                new_w = int(W * scale)
                new_h = int(H * scale)
                
                if new_w > 0 and new_h > 0:
                    # Resize source
                    # Add batch dim for interpolate
                    img_batch = img.unsqueeze(0)
                    scaled_img = torch.nn.functional.interpolate(img_batch, size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                    
                    # Create mask with feather
                    mask_small = _create_inner_feather_mask(new_w, new_h, feather_size).to(device)
                    
                    # Calculate placement
                    # Center
                    cx = canvas_width // 2
                    cy = canvas_height // 2
                    
                    # Top-left w.r.t center, plus offset
                    tl_x = cx - (new_w // 2) + translate_x
                    tl_y = cy - (new_h // 2) + translate_y
                    
                    # Clip to canvas bounds
                    x_start = max(0, tl_x)
                    y_start = max(0, tl_y)
                    x_end = min(canvas_width, tl_x + new_w)
                    y_end = min(canvas_height, tl_y + new_h)
                    
                    # Source offsets
                    src_x = x_start - tl_x
                    src_y = y_start - tl_y
                    w_slice = x_end - x_start
                    h_slice = y_end - y_start
                    
                    if w_slice > 0 and h_slice > 0:
                        # Composite
                        # We use simple copy for 2D composite on empty canvas
                        # But handle feather alpha
                        canvas[:, y_start:y_end, x_start:x_end] = scaled_img[:, src_y:src_y+h_slice, src_x:src_x+w_slice]
                        alpha[:, y_start:y_end, x_start:x_end] = mask_small[src_y:src_y+h_slice, src_x:src_x+w_slice].unsqueeze(0)
                
                # For output, we want RGB + Alpha channel if possible?
                # User asked for "transparent canvas". ComfyUI IMAGE is usually RGB.
                # But creating a MASK output allows standard inpainting.
                # However, if we return IMAGE with alpha, some nodes handle it.
                # We'll return RGB (premultiplied or masked?) and explicit MASK.
                # MASK: 1 = masked (keep), 0 = unmasked (inpaint).
                # Actually, standard Inpainting masks: 1 = inpaint this, 0 = keep this.
                # "transparent canvas" -> we want to inpaint the Transparent parts.
                # So Mask = 1 where Alpha = 0. Mask = 0 where Alpha = 1.
                # "alpha feather feather controls on the add flat image"
                # If we feather the image, the semi-transparent parts should be partially inpainted?
                # Usually: Mask = 1 - Alpha.
                
                out_img = canvas.permute(1, 2, 0) # BHWC
                out_mask = (1.0 - alpha).squeeze(0) # BHW
                
            else: # perspective
                # Perspective projection with proper scale behavior:
                # 
                # The `fov` parameter controls the projection distortion (how "wide" the lens is).
                # The `scale` parameter controls the projected image SIZE on the canvas.
                #
                # To scale the size WITHOUT increasing distortion, we scale the source image
                # dimensions and keep FOV constant. A larger source image projected at the
                # same FOV will cover more of the panorama.
                
                # Scale the source image first if scale != 1
                if abs(scale - 1.0) > 0.001:
                    scaled_h = int(H * scale)
                    scaled_w = int(W * scale)
                    if scaled_h > 0 and scaled_w > 0:
                        img_batch = img.unsqueeze(0)
                        img_scaled = torch.nn.functional.interpolate(
                            img_batch, size=(scaled_h, scaled_w), 
                            mode='bilinear', align_corners=False
                        ).squeeze(0)
                    else:
                        img_scaled = img
                else:
                    img_scaled = img
                
                # Use fov directly (no scaling) - controls distortion
                eff_fov = max(5.0, min(150.0, fov))
                
                # Scale feather proportionally
                scaled_feather = int(feather_size * scale) if scale > 1.0 else feather_size
                
                # Project
                proj_img, proj_mask = _perspective_insert(
                    img_scaled, 
                    canvas_height, canvas_width,
                    yaw, pitch, 0.0, # roll not exposed/default 0
                    eff_fov,
                    feather=scaled_feather
                )
                
                out_img = proj_img.permute(1, 2, 0)
                out_mask = (1.0 - proj_mask).squeeze(0)
            
            out_images.append(out_img)
            out_masks.append(out_mask)
            
        final_images = torch.stack(out_images)
        final_masks = torch.stack(out_masks)
        
        return (final_images, final_masks, context)


class LatLongOutpaintStitch:
    """Stitch high-res source back over generated equirectangular result."""
    
    DESCRIPTION = "Composite the original high-res source back over the generated result."
    CATEGORY = "LatLong/Outpaint"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_equirectangular",)
    FUNCTION = "stitch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_equirect": ("IMAGE", {"tooltip": "The generated/outpainted result."}),
                "stitch_context": ("STITCH_CONTEXT", {"tooltip": "Context from Setup node."}),
                "blend_mode": (["alpha", "overlay", "hard"], {"default": "alpha"}),
            }
        }

    def stitch(self, generated_equirect, stitch_context, blend_mode):
        # Unpack context
        ctx = stitch_context
        orig_flat = ctx["original_image"] # (N, H, W, C)
        mode = ctx["placement_mode"]
        scale = ctx["scale"]
        tx = ctx["translate_x"]
        ty = ctx["translate_y"]
        yaw = ctx["yaw"]
        pitch = ctx["pitch"]
        fov = ctx["fov"]
        feather = ctx["feather_size"]
        setup_w = ctx["setup_canvas_width"]
        setup_h = ctx["setup_canvas_height"]
        
        N, H_gen, W_gen, C_gen = generated_equirect.shape
        device = generated_equirect.device
        
        # Debug info
        print(f"[Stitch Debug] Generated equirect shape: {generated_equirect.shape}")
        print(f"[Stitch Debug] Original flat shape: {orig_flat.shape}")
        print(f"[Stitch Debug] Setup canvas: {setup_w}x{setup_h}")
        print(f"[Stitch Debug] Mode: {mode}, Scale: {scale}, Feather: {feather}")
        
        # Determine scale factor between setup canvas and generation
        # (e.g. if user upscaled the latent)
        res_scale = W_gen / float(setup_w)
        
        # We need to re-generate the composite layer at the NEW resolution
        # using the ORIGINAL high-res image.
        
        # Assume batch size 1 for context logic reuse, or match N
        # If N > 1, apply same context to all? Yes.
        
        final_images = []
        
        for i in range(N):
            gen_img = generated_equirect[i].permute(2, 0, 1) # (C, H, W)
            orig_src = orig_flat[i % len(orig_flat)].permute(2, 0, 1) # Loop if batches mismatch
            
            src_c, src_h, src_w = orig_src.shape
            
            if mode == "2d_composite":
                # Create empty canvas at CURRENT gen resolution
                canvas = torch.zeros_like(gen_img)
                alpha = torch.zeros((1, H_gen, W_gen), device=device)
                
                # Scale params by res_scale
                # scale param in 2d is pixel multiplier relative to source.
                # Wait, in Setup: "new_w = int(W * scale)".
                # The 'scale' param is relative to the SOURCE image pixels.
                # If we want to maintain the same relative size on the NEW canvas:
                # The "size on canvas" in Setup was (W * scale).
                # The "size on canvas" in Stitch should be (W * scale) * res_scale.
                
                # Wait, proper logic:
                # We want the flat image to occupy the same PROPORTION of the canvas.
                # Setup: occupied (W*scale) / setup_w fraction.
                # Stitch: should occupy (W_new) / W_gen fraction.
                # So W_new / W_gen = (W*scale) / setup_w
                # => W_new = (W*scale) * (W_gen / setup_w) = (W*scale) * res_scale.
                
                target_w = int(src_w * scale * res_scale)
                target_h = int(src_h * scale * res_scale)
                
                # Scaled feather
                target_feather = int(feather * res_scale)
                
                print(f"[Stitch Debug] 2D: target_w={target_w}, target_h={target_h}, target_feather={target_feather}, res_scale={res_scale}")
                
                if target_w > 0 and target_h > 0:
                     # Resize high-res source to target size
                     src_batch = orig_src.unsqueeze(0)
                     scaled_src = torch.nn.functional.interpolate(src_batch, size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
                     
                     mask_small = _create_inner_feather_mask(target_w, target_h, target_feather).to(device)
                     
                     print(f"[Stitch Debug] 2D: mask_small shape={mask_small.shape}, min={mask_small.min():.3f}, max={mask_small.max():.3f}")
                     
                     # Calculate placement
                     cx = W_gen // 2
                     cy = H_gen // 2
                     
                     # Scaled offsets
                     eff_Tx = int(tx * res_scale)
                     eff_Ty = int(ty * res_scale)
                     
                     tl_x = cx - (target_w // 2) + eff_Tx
                     tl_y = cy - (target_h // 2) + eff_Ty
                     
                     # Clip
                     x_start = max(0, tl_x)
                     y_start = max(0, tl_y)
                     x_end = min(W_gen, tl_x + target_w)
                     y_end = min(H_gen, tl_y + target_h)
                     
                     src_x = x_start - tl_x
                     src_y = y_start - tl_y
                     w_slice = x_end - x_start
                     h_slice = y_end - y_start
                     
                     print(f"[Stitch Debug] 2D: placement x={x_start}:{x_end}, y={y_start}:{y_end}, w_slice={w_slice}, h_slice={h_slice}")
                     
                     if w_slice > 0 and h_slice > 0:
                         canvas[:, y_start:y_end, x_start:x_end] = scaled_src[:, src_y:src_y+h_slice, src_x:src_x+w_slice]
                         alpha[:, y_start:y_end, x_start:x_end] = mask_small[src_y:src_y+h_slice, src_x:src_x+w_slice].unsqueeze(0)
                         print(f"[Stitch Debug] 2D: alpha after placement min={alpha.min():.3f}, max={alpha.max():.3f}, sum={alpha.sum():.0f}")
                     else:
                         print("[Stitch Debug] 2D: w_slice or h_slice is 0, no placement")
                else:
                     print("[Stitch Debug] 2D: target_w or target_h is 0, skipping")
            else: # perspective
                # Scale the source image to control size (same approach as Setup)
                # This avoids distortion from increasing FOV
                src_c, src_h, src_w = orig_src.shape
                
                if abs(scale - 1.0) > 0.001:
                    scaled_h = int(src_h * scale)
                    scaled_w = int(src_w * scale)
                    if scaled_h > 0 and scaled_w > 0:
                        src_batch = orig_src.unsqueeze(0)
                        src_scaled = torch.nn.functional.interpolate(
                            src_batch, size=(scaled_h, scaled_w),
                            mode='bilinear', align_corners=False
                        ).squeeze(0)
                    else:
                        src_scaled = orig_src
                else:
                    src_scaled = orig_src
                
                # Use fov directly - controls distortion
                eff_fov = max(5.0, min(150.0, fov))
                
                # Scale feather proportionally
                scaled_feather = int(feather * scale) if scale > 1.0 else feather
                
                proj_img, proj_mask = _perspective_insert(
                    src_scaled,
                    H_gen, W_gen,
                    yaw, pitch, 0.0,
                    eff_fov,
                    feather=scaled_feather
                )
                canvas = proj_img
                alpha = proj_mask
            
            # Composite
            # Result = Alpha * Source + (1 - Alpha) * Gen
            # (Assuming alpha blend)
            
            # gen_img is (C, H, W)
            # canvas is (C, H, W)
            # alpha is (1, H, W)
            
            final = alpha * canvas + (1.0 - alpha) * gen_img
            print(f"[Stitch Debug] Frame {i}: gen_img shape: {gen_img.shape}, canvas shape: {canvas.shape}, alpha shape: {alpha.shape}, final shape: {final.shape}")
            final_images.append(final.permute(1, 2, 0)) # BHWC
            
        result = torch.stack(final_images)
        print(f"[Stitch Debug] Final result shape: {result.shape}")
        return (result,)
