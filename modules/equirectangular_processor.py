import numpy as np
import torch
import cv2
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates


class EquirectangularProcessor:
    def __init__(self):
        pass
    
    @staticmethod
    def equirectangular_to_spherical(x: np.ndarray, y: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert equirectangular coordinates to spherical coordinates (lat, lon)"""
        lon = (x / width) * 2 * np.pi - np.pi  # -π to π
        lat = np.pi/2 - (y / height) * np.pi    # π/2 to -π/2
        return lat, lon
    
    @staticmethod
    def spherical_to_cartesian(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert spherical coordinates to 3D cartesian coordinates"""
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return x, y, z
    
    @staticmethod
    def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 3D cartesian coordinates to spherical coordinates"""
        lat = np.arcsin(np.clip(z, -1, 1))
        lon = np.arctan2(y, x)
        return lat, lon
    
    @staticmethod
    def spherical_to_equirectangular(lat: np.ndarray, lon: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert spherical coordinates to equirectangular coordinates"""
        # Normalize longitude to [0, 2π] then to [0, width]
        lon_norm = (lon + np.pi) / (2 * np.pi)
        x = lon_norm * width
        
        # Convert latitude from [π/2, -π/2] to [0, height]
        lat_norm = (np.pi/2 - lat) / np.pi
        y = lat_norm * height
        
        return x, y
    
    @staticmethod
    def create_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Create 3D rotation matrix from yaw, pitch, roll angles in degrees"""
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Using scipy for robust rotation matrix creation
        r = Rotation.from_euler('zyx', [yaw_rad, pitch_rad, roll_rad])
        return r.as_matrix()
    
    @staticmethod
    def apply_rotation(x: np.ndarray, y: np.ndarray, z: np.ndarray, rotation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply rotation matrix to 3D points"""
        points = np.stack([x, y, z], axis=-1)
        original_shape = points.shape[:-1]
        points_flat = points.reshape(-1, 3)
        
        rotated_points = points_flat @ rotation_matrix.T
        rotated_points = rotated_points.reshape(*original_shape, 3)
        
        return rotated_points[..., 0], rotated_points[..., 1], rotated_points[..., 2]
    
    @staticmethod
    def interpolate_image(image: np.ndarray, x: np.ndarray, y: np.ndarray, method: str = 'lanczos') -> np.ndarray:
        """Perform high-quality interpolation on image at given coordinates"""
        height, width = image.shape[:2]
        
        # Handle different interpolation methods
        if method == 'nearest':
            order = 0
        elif method == 'bilinear':
            order = 1
        elif method == 'bicubic':
            order = 3
        elif method == 'lanczos':
            # Use scipy's map_coordinates with spline interpolation for high quality
            order = 3
        else:
            order = 3  # Default to bicubic
        
        # Clamp coordinates to valid range
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        
        if len(image.shape) == 3:  # Color image
            # Process each channel separately
            interpolated = np.zeros_like(image)
            for c in range(image.shape[2]):
                interpolated[..., c] = map_coordinates(
                    image[..., c], 
                    [y, x], 
                    order=order, 
                    mode='wrap',  # Wrap for equirectangular longitude wrapping
                    prefilter=True
                )
        else:  # Grayscale image
            interpolated = map_coordinates(
                image, 
                [y, x], 
                order=order, 
                mode='wrap',
                prefilter=True
            )
        
        return interpolated
    
    @staticmethod
    def bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Legacy bilinear interpolation method - kept for compatibility"""
        return EquirectangularProcessor.interpolate_image(image, x, y, method='bilinear')
    
    @classmethod
    def rotate_equirectangular(cls, image: np.ndarray, yaw: float = 0, pitch: float = 0, roll: float = 0, horizon_offset: float = 0, interpolation: str = 'lanczos') -> np.ndarray:
        """Rotate an equirectangular image with optional horizon adjustment"""
        height, width = image.shape[:2]
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to spherical coordinates
        lat, lon = cls.equirectangular_to_spherical(x_coords, y_coords, width, height)
        
        # Apply horizon offset
        lat += np.radians(horizon_offset)
        lat = np.clip(lat, -np.pi/2, np.pi/2)  # Clamp to valid latitude range
        
        # Convert to 3D cartesian
        x_3d, y_3d, z_3d = cls.spherical_to_cartesian(lat, lon)
        
        # Create and apply rotation matrix
        rotation_matrix = cls.create_rotation_matrix(yaw, pitch, roll)
        x_rot, y_rot, z_rot = cls.apply_rotation(x_3d, y_3d, z_3d, rotation_matrix)
        
        # Convert back to spherical
        lat_rot, lon_rot = cls.cartesian_to_spherical(x_rot, y_rot, z_rot)
        
        # Convert back to equirectangular coordinates
        x_new, y_new = cls.spherical_to_equirectangular(lat_rot, lon_rot, width, height)
        
        # Handle longitude wrapping
        x_new = np.mod(x_new, width)
        
        # Interpolate to get final image with specified method
        rotated_image = cls.interpolate_image(image, x_new, y_new, method=interpolation)
        
        return rotated_image
    
    @staticmethod
    def crop_to_180(image: np.ndarray, output_width: Optional[int] = None, output_height: Optional[int] = None, interpolation: str = 'lanczos') -> np.ndarray:
        """Crop equirectangular image to 180 degree field of view"""
        height, width = image.shape[:2]
        
        # Calculate crop boundaries (center 180 degrees out of 360)
        start_x = width // 4  # Start at 90 degrees
        end_x = start_x + width // 2  # End at 270 degrees (180 degree span)
        
        # Crop the image
        cropped = image[:, start_x:end_x]
        
        # Resize if output dimensions specified with high-quality interpolation
        if output_width is not None and output_height is not None:
            # Map interpolation methods to OpenCV constants
            if interpolation == 'nearest':
                cv_interp = cv2.INTER_NEAREST
            elif interpolation == 'bilinear':
                cv_interp = cv2.INTER_LINEAR
            elif interpolation == 'bicubic':
                cv_interp = cv2.INTER_CUBIC
            elif interpolation == 'lanczos':
                cv_interp = cv2.INTER_LANCZOS4
            else:
                cv_interp = cv2.INTER_LANCZOS4  # Default to highest quality
            
            cropped = cv2.resize(cropped, (output_width, output_height), interpolation=cv_interp)
        
        return cropped
    
    @staticmethod
    def crop_to_square(image: np.ndarray, interpolation: str = 'lanczos') -> np.ndarray:
        """Crop equirectangular image to square (width = height of original image)"""
        height, width = image.shape[:2]
        
        # Calculate crop boundaries to get a square with width = original height
        crop_width = height
        start_x = (width - crop_width) // 2  # Center the crop
        end_x = start_x + crop_width
        
        # Ensure we don't exceed image boundaries
        start_x = max(0, start_x)
        end_x = min(width, end_x)
        
        # Crop the image to square
        cropped = image[:, start_x:end_x]
        
        return cropped
    
    @classmethod
    def process_equirectangular(cls, 
                              image: np.ndarray,
                              yaw: float = 0,
                              pitch: float = 0, 
                              roll: float = 0,
                              horizon_offset: float = 0,
                              crop_to_180: bool = False,
                              crop_to_square: bool = False,
                              output_width: Optional[int] = None,
                              output_height: Optional[int] = None,
                              interpolation: str = 'lanczos') -> np.ndarray:
        """Main processing function that combines all operations"""
        
        # Apply rotation and horizon adjustment
        if yaw != 0 or pitch != 0 or roll != 0 or horizon_offset != 0:
            processed_image = cls.rotate_equirectangular(image, yaw, pitch, roll, horizon_offset, interpolation)
        else:
            processed_image = image.copy()
        
        # Apply cropping if requested (square takes precedence over 180)
        if crop_to_square:
            processed_image = cls.crop_to_square(processed_image, interpolation)
        elif crop_to_180:
            processed_image = cls.crop_to_180(processed_image, output_width, output_height, interpolation)
        
        return processed_image