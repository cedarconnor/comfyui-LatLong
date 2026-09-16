import numpy as np
import torch
import cv2
from typing import Tuple, Optional, List
from .sampling import remap, torch_remap
from scipy.spatial.transform import Rotation
import gc


class EquirectangularProcessor:
    def __init__(self):
        pass

    @staticmethod
    def calculate_optimal_tile_size(image_width: int, image_height: int,
                                     max_memory_mb: float = 2048) -> Tuple[int, int]:
        """Calculate optimal tile size for processing large images.

        Args:
            image_width: Width of the image
            image_height: Height of the image
            max_memory_mb: Maximum memory per tile in megabytes

        Returns:
            Tuple of (tile_height, tile_width) in pixels
        """
        # Estimate memory per pixel (3 channels * 4 bytes float32 + overhead)
        bytes_per_pixel = 3 * 4 * 2  # *2 for input + output buffers
        max_pixels = (max_memory_mb * 1024 * 1024) / bytes_per_pixel

        # For 16K images (16384x8192), aim for 4-16 tiles
        # Calculate square-ish tiles that fit within memory constraints
        if image_width * image_height <= max_pixels:
            # Image fits in memory, no tiling needed
            return (image_height, image_width)

        # Calculate number of tiles needed
        num_tiles = int(np.ceil((image_width * image_height) / max_pixels))
        tiles_per_row = int(np.ceil(np.sqrt(num_tiles)))

        tile_height = image_height // tiles_per_row
        tile_width = image_width // tiles_per_row

        return (tile_height, tile_width)

    @staticmethod
    def should_use_tiled_processing(image_width: int, image_height: int,
                                     tile_threshold_px: int = 8192) -> bool:
        """Determine if tiled processing should be used.

        Args:
            image_width: Width of the image
            image_height: Height of the image
            tile_threshold_px: Threshold dimension to trigger tiling

        Returns:
            True if tiling should be used
        """
        return image_width > tile_threshold_px or image_height > tile_threshold_px or image_width * image_height > 1048576

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
    def interpolate_image(image, x, y, method='lanczos'):
        return remap(image, x, y, method)


    @staticmethod
    def bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Legacy bilinear interpolation method - kept for compatibility"""
        return EquirectangularProcessor.interpolate_image(image, x, y, method='bilinear')

    @classmethod
    def rotate_equirectangular(cls, image: np.ndarray, yaw: float = 0, pitch: float = 0, roll: float = 0,
                              horizon_offset: float = 0, interpolation: str = 'lanczos',
                              use_tiling: Optional[bool] = None, tile_size: int = 2048,
                              overlap: int = 64, progress_callback=None) -> np.ndarray:
        """Rotate an equirectangular image with optional horizon adjustment.

        Args:
            image: Input equirectangular image
            yaw, pitch, roll: Rotation angles in degrees
            horizon_offset: Vertical horizon shift in degrees
            interpolation: Interpolation method
            use_tiling: Force tiling on/off. If None, auto-decide based on image size
            tile_size: Size of tiles for tiled processing
            overlap: Pixel overlap between tiles to avoid seams

        Returns:
            Rotated equirectangular image
        """
        height, width = image.shape[:2]
        if pitch == roll == horizon_offset == 0 and np.isclose(yaw * width / 360, round(yaw * width / 360), atol=1e-6, rtol=0):
            return np.roll(image, -round(yaw * width / 360), axis=1).copy()

        # Decide whether to use tiling
        if use_tiling is None:
            use_tiling = cls.should_use_tiled_processing(width, height)

        if not use_tiling:
            # Original non-tiled implementation
            return cls._rotate_equirectangular_direct(image, yaw, pitch, roll,
                                                     horizon_offset, interpolation)
        else:
            # Tiled implementation for large images
            return cls._rotate_equirectangular_tiled(image, yaw, pitch, roll,
                                                    horizon_offset, interpolation,
                                                    tile_size, overlap, progress_callback)

    @classmethod
    def _rotate_equirectangular_direct(cls, image: np.ndarray, yaw: float, pitch: float,
                                       roll: float, horizon_offset: float,
                                       interpolation: str) -> np.ndarray:
        """Direct (non-tiled) rotation implementation."""
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

        # Interpolate to get final image with specified method (interpolator wraps horizontally)
        rotated_image = cls.interpolate_image(image, x_new, y_new, method=interpolation)

        return rotated_image

    @classmethod
    def _rotate_equirectangular_tiled(cls, image: np.ndarray, yaw: float, pitch: float,
                                     roll: float, horizon_offset: float, interpolation: str,
                                     tile_size: int, overlap: int, progress_callback=None) -> np.ndarray:
        """Tiled rotation implementation for large images.

        Processes the image in horizontal bands to reduce memory usage.
        Each band has an overlap region to avoid seam artifacts.
        """
        height, width = image.shape[:2]
        result = np.zeros_like(image)

        # Pre-compute rotation matrix once
        rotation_matrix = cls.create_rotation_matrix(yaw, pitch, roll)

        # Process in horizontal bands
        tile_size = max(1, min(tile_size, 262144 // max(1, width)))
        overlap = 0  # Independent inverse samples need no overlap.
        num_bands = int(np.ceil(height / tile_size))

        for band_idx in range(num_bands):
            if progress_callback is not None:
                progress_callback()
            # Calculate band boundaries with overlap
            y_start = max(0, band_idx * tile_size - overlap)
            y_end = min(height, (band_idx + 1) * tile_size + overlap)
            band_height = y_end - y_start

            # Create coordinate grids for this band only
            x_coords, y_coords = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(y_start, y_end, dtype=np.float64))

            # Convert to spherical coordinates
            lat, lon = cls.equirectangular_to_spherical(x_coords, y_coords, width, height)

            # Apply horizon offset
            lat += np.radians(horizon_offset)
            lat = np.clip(lat, -np.pi/2, np.pi/2)

            # Convert to 3D cartesian
            x_3d, y_3d, z_3d = cls.spherical_to_cartesian(lat, lon)

            # Apply pre-computed rotation
            x_rot, y_rot, z_rot = cls.apply_rotation(x_3d, y_3d, z_3d, rotation_matrix)

            # Convert back to spherical
            lat_rot, lon_rot = cls.cartesian_to_spherical(x_rot, y_rot, z_rot)

            # Convert back to equirectangular coordinates
            x_new, y_new = cls.spherical_to_equirectangular(lat_rot, lon_rot, width, height)

            # Interpolate this band
            band_rotated = cls.interpolate_image(image, x_new, y_new, method=interpolation)

            # Calculate the actual region to copy (excluding overlap margins)
            if band_idx == 0:
                # First band: no overlap at top
                result_y_start = 0
                band_y_start = 0
                result_y_end = min(tile_size, height)
                band_y_end = result_y_end - y_start
            elif band_idx == num_bands - 1:
                # Last band: no overlap at bottom
                result_y_start = band_idx * tile_size
                band_y_start = result_y_start - y_start
                result_y_end = height
                band_y_end = band_height
            else:
                # Middle bands: use center region, avoiding overlaps
                result_y_start = band_idx * tile_size
                band_y_start = overlap
                result_y_end = min((band_idx + 1) * tile_size, height)
                band_y_end = band_y_start + (result_y_end - result_y_start)

            # Copy the band to result
            result[result_y_start:result_y_end, :] = band_rotated[band_y_start:band_y_end, :]

            # Free memory
            del x_coords, y_coords, lat, lon, x_3d, y_3d, z_3d
            del x_rot, y_rot, z_rot, lat_rot, lon_rot, x_new, y_new, band_rotated

        return result

    @staticmethod
    def crop_to_180(image: np.ndarray,
                    output_width: Optional[int] = None,
                    output_height: Optional[int] = None,
                    interpolation: str = 'lanczos',
                    center_longitude_deg: float = 0.0,
                    fov_degrees: float = 180.0) -> np.ndarray:
        """Crop equirectangular image to a longitude span (default 180°) centered at a longitude.

        Args:
            image: Input equirectangular image (H, W, C) in [0,1] float or uint8.
            output_width: Optional output width for resize.
            output_height: Optional output height for resize.
            interpolation: Interpolation method for resize.
            center_longitude_deg: Center longitude in degrees (-180..180).
            fov_degrees: Horizontal field of view in degrees (0..360].
        """
        height, width = image.shape[:2]

        # Compute crop width in pixels based on FOV
        crop_width = max(1, int(round(width * (float(fov_degrees) / 360.0))))
        # Center x position in pixels
        center_x = (float(center_longitude_deg) + 180.0) / 360.0 * width
        start_x = int(np.floor(center_x - crop_width / 2.0))
        end_x = start_x + crop_width

        # Seam-safe horizontal crop via concatenation
        image_pad = np.concatenate([image, image, image], axis=1)
        start_x_pad = start_x + width
        end_x_pad = start_x_pad + crop_width
        cropped = image_pad[:, start_x_pad:end_x_pad]

        # Resize if output dimensions specified with high-quality interpolation
        if output_width is not None and output_height is not None:
            if interpolation == 'nearest':
                cv_interp = cv2.INTER_NEAREST
            elif interpolation == 'bilinear':
                cv_interp = cv2.INTER_LINEAR
            elif interpolation == 'bicubic':
                cv_interp = cv2.INTER_CUBIC
            elif interpolation == 'lanczos':
                cv_interp = cv2.INTER_LANCZOS4
            else:
                cv_interp = cv2.INTER_LANCZOS4
            cropped = cv2.resize(cropped, (int(output_width), int(output_height)), interpolation=cv_interp)
            if image.ndim == 3 and cropped.ndim == 2:
                cropped = cropped[..., None]

        return cropped

    @staticmethod
    def _cached_perspective_maps(in_width, in_height, out_width, out_height, yaw, pitch, roll, fov_degrees):
        if not np.isfinite([yaw, pitch, roll, fov_degrees]).all() or not 0 < fov_degrees < 180:
            raise ValueError("Projection angles must be finite and FOV must be between 0 and 180 degrees")
        if min(in_width, in_height, out_width, out_height) < 1:
            raise ValueError("Projection dimensions must be positive")
        # Kept as a callable name for existing callers; maps no longer persist between executions.
        u, v = np.meshgrid(np.arange(out_width, dtype=np.float32) + 0.5,
                           np.arange(out_height, dtype=np.float32) + 0.5)
        f = out_width / (2 * np.tan(np.radians(fov_degrees) / 2))
        directions = np.stack((np.ones_like(u), (u - out_width / 2) / f,
                               -(v - out_height / 2) / f), axis=-1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = directions @ EquirectangularProcessor.create_rotation_matrix(yaw, pitch, roll).T
        lat, lon = EquirectangularProcessor.cartesian_to_spherical(*np.moveaxis(directions, -1, 0))
        return EquirectangularProcessor.spherical_to_equirectangular(lat, lon, in_width, in_height)


    @classmethod
    def perspective_extract(cls, image, out_width, out_height, yaw=0., pitch=0., roll=0., fov_degrees=90., interpolation='lanczos'):
        maps = cls._cached_perspective_maps(image.shape[1], image.shape[0], out_width, out_height, yaw, pitch, roll, fov_degrees)
        return cls.interpolate_image(image, *maps, method=interpolation)


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

    # =============================
    # Torch (GPU) implementations
    # =============================
    @staticmethod
    def _torch_rotation_matrix(yaw, pitch, roll, device, dtype):
        return torch.as_tensor(EquirectangularProcessor.create_rotation_matrix(yaw, pitch, roll), device=device, dtype=dtype)


    @classmethod
    def torch_rotate_equirectangular(cls, image, yaw=0., pitch=0., roll=0., horizon_offset=0., interpolation='bilinear', tile_size=256, progress_callback=None):
        h, w, c = image.shape
        if pitch == roll == horizon_offset == 0 and np.isclose(yaw * w / 360, round(yaw * w / 360), atol=1e-6, rtol=0):
            return torch.roll(image, -round(yaw * w / 360), dims=1).clone()
        result = torch.empty_like(image)
        R = cls._torch_rotation_matrix(yaw, pitch, roll, image.device, image.dtype)
        padded = torch.cat((image[:, -1:], image, image[:, :1]), dim=1)
        for start in range(0, h, tile_size):
            ys, xs = torch.meshgrid(torch.arange(start, min(h, start + tile_size), device=image.device, dtype=image.dtype),
                                   torch.arange(w, device=image.device, dtype=image.dtype), indexing='ij')
            lon = xs / w * (2 * np.pi) - np.pi
            lat = (np.pi / 2 - ys / h * np.pi + np.radians(horizon_offset)).clamp(-np.pi/2, np.pi/2)
            xyz = torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), -1) @ R.T
            src_x = (torch.atan2(xyz[..., 1], xyz[..., 0]) + np.pi) / (2 * np.pi) * w
            src_y = (np.pi / 2 - torch.asin(xyz[..., 2].clamp(-1, 1))) / np.pi * h
            result[start:start + tile_size] = torch_remap(image, src_x, src_y, interpolation, padded=padded)
            if progress_callback is not None:
                progress_callback()
        return result


    @classmethod
    def torch_perspective_extract(cls, image, out_width, out_height, yaw=0., pitch=0., roll=0., fov_degrees=90., interpolation='bilinear'):
        maps = cls._cached_perspective_maps(image.shape[1], image.shape[0], out_width, out_height, yaw, pitch, roll, fov_degrees)
        x, y = (torch.as_tensor(m, device=image.device, dtype=image.dtype) for m in maps)
        return torch_remap(image, x, y, interpolation)


    # =============================
    # Cubemap generation
    # =============================
    @staticmethod
    def _cube_face_direction(face: str, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map face coordinates (u,v in [-1,1]) to direction vector (x,y,z) in our coordinate system.

        Axes: x: lon 0°, y: lon 90°, z: +lat (up).
        """
        if face == 'front':  # +X
            x = np.ones_like(u)
            y = u
            z = v
        elif face == 'back':  # -X
            x = -np.ones_like(u)
            y = -u
            z = v
        elif face == 'right':  # +Y
            x = -u
            y = np.ones_like(u)
            z = v
        elif face == 'left':  # -Y
            x = u
            y = -np.ones_like(u)
            z = v
        elif face == 'top':  # +Z
            x = u
            y = v
            z = np.ones_like(u)
        elif face == 'bottom':  # -Z
            x = u
            y = -v
            z = -np.ones_like(u)
        else:
            raise ValueError(f"Unknown face: {face}")
        # Normalize
        norm = np.sqrt(x*x + y*y + z*z)
        return x / norm, y / norm, z / norm

    @classmethod
    def equirectangular_to_cubemap(cls,
                                   image: np.ndarray,
                                   face_size: int = 512,
                                   layout: str = '3x2',
                                   interpolation: str = 'lanczos') -> np.ndarray:
        """Generate a cubemap atlas from an equirectangular image.

        Layout '3x2' order: [left, front, right] on top row; [back, top, bottom] bottom row.
        Returns atlas image with shape (2*face_size, 3*face_size, C).
        """
        h, w = image.shape[:2]
        # Interp mapping
        if interpolation == 'nearest': cv_interp = cv2.INTER_NEAREST
        elif interpolation == 'bilinear': cv_interp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic': cv_interp = cv2.INTER_CUBIC
        else: cv_interp = cv2.INTER_LANCZOS4

        faces = ['left', 'front', 'right', 'back', 'top', 'bottom']
        # Prepare atlas
        atlas = np.zeros((face_size * 2, face_size * 3, image.shape[2]), dtype=image.dtype)

        def render_face(face: str) -> np.ndarray:
            xs = np.linspace(-1 + 1/face_size, 1 - 1/face_size, face_size, dtype=np.float32)
            ys = np.linspace(-1 + 1/face_size, 1 - 1/face_size, face_size, dtype=np.float32)
            u, v = np.meshgrid(xs, ys)
            x, y, z = cls._cube_face_direction(face, u, v)
            lat = np.arcsin(np.clip(z, -1.0, 1.0))
            lon = np.arctan2(y, x)
            x_eq, y_eq = cls.spherical_to_equirectangular(lat, lon, w, h)
            map_x = np.mod(x_eq, w).astype(np.float32)
            map_y = np.clip(y_eq, 0, h - 1).astype(np.float32)
            face_img = cls.interpolate_image(image, map_x, map_y, method=interpolation)
            return face_img

        # Place faces
        atlas[0:face_size, 0:face_size] = render_face('left')
        atlas[0:face_size, face_size:2*face_size] = render_face('front')
        atlas[0:face_size, 2*face_size:3*face_size] = render_face('right')
        atlas[face_size:2*face_size, 0:face_size] = render_face('back')
        atlas[face_size:2*face_size, face_size:2*face_size] = render_face('top')
        atlas[face_size:2*face_size, 2*face_size:3*face_size] = render_face('bottom')

        return atlas

    @classmethod
    def cubemap_to_equirectangular(cls, cubemap_atlas, output_width=2048, output_height=1024, layout='3x2', interpolation='lanczos'):
        h, w = cubemap_atlas.shape[:2]
        fs = h // 2
        if layout != '3x2' or h != fs * 2 or w != fs * 3 or fs < 1:
            raise ValueError('Cubemap atlas must have six square faces in a 3x2 layout')
        faces = [cubemap_atlas[0:fs, fs:2*fs], cubemap_atlas[fs:2*fs, :fs],
                 cubemap_atlas[0:fs, 2*fs:3*fs], cubemap_atlas[0:fs, :fs],
                 cubemap_atlas[fs:2*fs, fs:2*fs], cubemap_atlas[fs:2*fs, 2*fs:3*fs]]
        output = np.empty((output_height, output_width, cubemap_atlas.shape[2]), dtype=cubemap_atlas.dtype)
        for start in range(0, output_height, 128):
            xx, yy = np.meshgrid(np.arange(output_width, dtype=np.float32), np.arange(start, min(start+128, output_height), dtype=np.float32))
            lat, lon = cls.equirectangular_to_spherical(xx, yy, output_width, output_height)
            x, y, z = cls.spherical_to_cartesian(lat, lon)
            major = np.argmax(np.stack((abs(x), abs(y), abs(z))), axis=0)
            band = output[start:start+128]
            definitions = [(major == 0) & (x >= 0), (major == 0) & (x < 0),
                           (major == 1) & (y >= 0), (major == 1) & (y < 0),
                           (major == 2) & (z >= 0), (major == 2) & (z < 0)]
            for idx, mask in enumerate(definitions):
                if not np.any(mask):
                    continue
                xm, ym, zm = x[mask], y[mask], z[mask]
                if idx == 0: u, v = ym / xm, zm / xm
                elif idx == 1: u, v = ym / xm, zm / -xm
                elif idx == 2: u, v = -xm / ym, zm / ym
                elif idx == 3: u, v = xm / -ym, zm / -ym
                elif idx == 4: u, v = xm / zm, ym / zm
                else: u, v = xm / -zm, -ym / -zm
                band[mask] = remap(faces[idx], (u+1)*fs/2 - .5, (v+1)*fs/2 - .5, interpolation, wrap_x=False)
        return output


    @staticmethod
    def mirror_flip_equirectangular(image: np.ndarray,
                                    mirror_horizontal: bool = False,
                                    mirror_vertical: bool = False) -> np.ndarray:
        """Mirror/flip an equirectangular image with proper spherical wrapping.

        Args:
            image: Input equirectangular image.
            mirror_horizontal: Flip left-right (longitude flip).
            mirror_vertical: Flip top-bottom (latitude flip).

        Returns:
            Flipped equirectangular image.
        """
        result = image.copy()

        if mirror_horizontal:
            # Horizontal flip: reverse longitude (flip left-right)
            result = np.fliplr(result)

        if mirror_vertical:
            # Vertical flip: reverse latitude (flip top-bottom)
            result = np.flipud(result)

        return result

    @staticmethod
    def resize_equirectangular(image: np.ndarray,
                               output_width: Optional[int] = None,
                               output_height: Optional[int] = None,
                               maintain_aspect: bool = True,
                               interpolation: str = 'lanczos') -> np.ndarray:
        """Resize an equirectangular image with optional aspect ratio preservation.

        Args:
            image: Input equirectangular image.
            output_width: Target width (required if maintain_aspect=False).
            output_height: Target height (optional if maintain_aspect=True).
            maintain_aspect: Preserve 2:1 aspect ratio (standard for equirectangular).
            interpolation: Resampling method.

        Returns:
            Resized equirectangular image.
        """
        h, w = image.shape[:2]

        if maintain_aspect:
            # Standard equirectangular is 2:1 ratio
            if output_width is not None:
                out_w = output_width
                out_h = output_width // 2
            elif output_height is not None:
                out_h = output_height
                out_w = output_height * 2
            else:
                raise ValueError("Either output_width or output_height must be specified")
        else:
            if output_width is None or output_height is None:
                raise ValueError("Both output_width and output_height must be specified when maintain_aspect=False")
            out_w = output_width
            out_h = output_height

        # Map interpolation to OpenCV
        if interpolation == 'nearest':
            cv_interp = cv2.INTER_NEAREST
        elif interpolation == 'bilinear':
            cv_interp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            cv_interp = cv2.INTER_CUBIC
        elif interpolation == 'lanczos':
            cv_interp = cv2.INTER_LANCZOS4
        else:
            cv_interp = cv2.INTER_LANCZOS4

        resized = cv2.resize(image, (out_w, out_h), interpolation=cv_interp)
        if image.ndim == 3 and resized.ndim == 2:
            resized = resized[..., None]
        return resized

    @staticmethod
    def get_preset_rotation(preset: str) -> Tuple[float, float, float]:
        """Get yaw, pitch, roll values for preset rotation angles.

        Args:
            preset: One of 'front', 'back', 'left', 'right', 'up', 'down', 'custom'.

        Returns:
            Tuple of (yaw, pitch, roll) in degrees.
        """
        presets = {
            'front': (0.0, 0.0, 0.0),
            'back': (180.0, 0.0, 0.0),
            'left': (-90.0, 0.0, 0.0),
            'right': (90.0, 0.0, 0.0),
            'up': (0.0, 90.0, 0.0),
            'down': (0.0, -90.0, 0.0),
        }
        return presets.get(preset, (0.0, 0.0, 0.0))

    @staticmethod
    def blend_edges(image: np.ndarray, blend_width: int = 10,
                   mode: str = "cosine") -> np.ndarray:
        """Blend left and right edges for seamless wraparound.

        Creates a smooth transition between the left and right edges of a panorama
        to ensure seamless wraparound when viewed in 360° viewers.

        Args:
            image: Input image in (H, W, C) format
            blend_width: Width of blend region in pixels (default 10)
            mode: Blending function:
                - 'linear': Simple linear interpolation
                - 'cosine': Smooth cosine interpolation (recommended)
                - 'smooth': Quadratic smooth interpolation

        Returns:
            Image with blended edges

        Example:
            >>> panorama = np.random.rand(1024, 2048, 3)
            >>> blended = blend_edges(panorama, blend_width=20, mode="cosine")
            >>> # Left and right edges now transition smoothly
        """
        H, W = image.shape[:2]

        # Validate blend width
        if blend_width <= 0 or blend_width >= W // 2:
            print(f"Warning: blend_width {blend_width} invalid, must be 0 < width < {W//2}")
            return image

        # Extract edge regions
        left_edge = image[:, :blend_width, :].copy()
        right_edge = image[:, -blend_width:, :].copy()

        # Create blend weights based on mode
        if mode == "linear":
            # Simple linear ramp: 0 -> 1
            weights = np.linspace(0, 1, blend_width)

        elif mode == "cosine":
            # Smooth cosine curve: 0 -> 1
            # Uses (1 - cos(πx)) / 2 for smooth S-curve
            t = np.linspace(0, np.pi, blend_width)
            weights = (1 - np.cos(t)) / 2

        elif mode == "smooth":
            # Quadratic smooth: x²
            weights = np.linspace(0, 1, blend_width) ** 2

        else:
            raise ValueError(f"Unknown blend mode: {mode}. Use 'linear', 'cosine', or 'smooth'")

        # Reshape weights for broadcasting: (1, blend_width, 1)
        weights = weights.reshape(1, -1, 1)

        # Blend edges using weighted average
        # Left edge: transitions from left_edge to right_edge
        # Right edge: transitions from right_edge to left_edge
        target = (image[:, :1, :] + image[:, -1:, :]) * 0.5
        blended_left = left_edge + (target - left_edge[:, :1, :]) * (1 - weights)
        blended_right = right_edge + (target - right_edge[:, -1:, :]) * weights

        # Apply blending to image
        result = image.copy()
        result[:, :blend_width, :] = blended_left
        result[:, -blend_width:, :] = blended_right

        return result

    @staticmethod
    def check_edge_continuity(image: np.ndarray, threshold: float = 0.05) -> bool:
        """Check if left and right edges are continuous (for validation).

        Measures the average pixel difference between the leftmost and rightmost
        columns to determine if the panorama wraps seamlessly.

        Args:
            image: Input image in (H, W, C) format
            threshold: Maximum allowed difference (0-1 scale). Default 0.05 = 5%

        Returns:
            True if edges are continuous within threshold

        Example:
            >>> panorama = np.random.rand(1024, 2048, 3)
            >>> panorama = blend_edges(panorama)
            >>> check_edge_continuity(panorama)
            True
            >>> # Without blending, likely returns False
        """
        # Get leftmost and rightmost columns
        left_edge = image[:, 0, :]   # (H, C)
        right_edge = image[:, -1, :]  # (H, C)

        # Calculate mean absolute difference
        diff = np.abs(left_edge - right_edge).mean()

        return diff < threshold
