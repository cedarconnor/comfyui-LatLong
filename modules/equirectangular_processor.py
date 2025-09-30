import numpy as np
import torch
import cv2
from typing import Tuple, Optional
from functools import lru_cache
from scipy.spatial.transform import Rotation


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
        """Interpolate image at given coordinates using OpenCV remap.

        Notes:
        - We modulo-wrap longitude (x) to preserve equirectangular seam continuity.
        - We clamp latitude (y) to valid range to avoid pole artifacts.
        - Interpolation methods map to OpenCV equivalents with true Lanczos.
        """
        height, width = image.shape[:2]

        # Map interpolation methods to OpenCV constants
        if method == 'nearest':
            cv_interp = cv2.INTER_NEAREST
        elif method == 'bilinear':
            cv_interp = cv2.INTER_LINEAR
        elif method == 'bicubic':
            cv_interp = cv2.INTER_CUBIC
        elif method == 'lanczos':
            cv_interp = cv2.INTER_LANCZOS4
        else:
            cv_interp = cv2.INTER_CUBIC

        # Wrap x (longitude) and clamp y (latitude)
        x_wrapped = np.mod(x, width).astype(np.float32)
        y_clamped = np.clip(y, 0, height - 1).astype(np.float32)

        # OpenCV remap expects maps shaped (H, W) in float32
        map_x = x_wrapped
        map_y = y_clamped

        # remap handles multi-channel images directly
        interpolated = cv2.remap(image, map_x, map_y, interpolation=cv_interp, borderMode=cv2.BORDER_WRAP)
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
        
        # Interpolate to get final image with specified method (interpolator wraps horizontally)
        rotated_image = cls.interpolate_image(image, x_new, y_new, method=interpolation)

        return rotated_image
    
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

        return cropped

    @staticmethod
    @lru_cache(maxsize=16)
    def _cached_perspective_maps(in_width: int,
                                 in_height: int,
                                 out_width: int,
                                 out_height: int,
                                 yaw: float,
                                 pitch: float,
                                 roll: float,
                                 fov_degrees: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and cache remap grids (map_x, map_y) for perspective extraction."""
        # Output pixel grid
        xs = np.linspace(0.5, out_width - 0.5, out_width, dtype=np.float32)
        ys = np.linspace(0.5, out_height - 0.5, out_height, dtype=np.float32)
        u, v = np.meshgrid(xs, ys)  # shape (Hout, Wout)

        # Camera intrinsics from horizontal FOV
        f = (out_width / (2.0 * np.tan(np.radians(fov_degrees) / 2.0))).astype(np.float32) if isinstance(out_width, np.ndarray) else (out_width / (2.0 * np.tan(np.radians(fov_degrees) / 2.0)))
        cx = out_width / 2.0
        cy = out_height / 2.0

        x_cam = (u - cx) / f
        y_cam = (v - cy) / f
        z_cam = np.ones_like(x_cam, dtype=np.float32)

        # Normalize direction vectors
        norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
        x_cam /= norm
        y_cam /= norm
        z_cam /= norm

        # Apply camera rotation (yaw, pitch, roll)
        R = EquirectangularProcessor.create_rotation_matrix(yaw, pitch, roll)
        pts = np.stack([x_cam, y_cam, z_cam], axis=-1)
        orig_shape = pts.shape[:-1]
        pts_flat = pts.reshape(-1, 3)
        rot = pts_flat @ R.T
        rot = rot.reshape(*orig_shape, 3)

        x_r, y_r, z_r = rot[..., 0], rot[..., 1], rot[..., 2]
        lat = np.arcsin(np.clip(z_r, -1.0, 1.0))
        lon = np.arctan2(y_r, x_r)

        # Map to equirectangular pixel coordinates
        x_eq, y_eq = EquirectangularProcessor.spherical_to_equirectangular(lat, lon, in_width, in_height)
        x_eq = np.mod(x_eq, in_width).astype(np.float32)
        y_eq = np.clip(y_eq, 0, in_height - 1).astype(np.float32)
        return x_eq, y_eq

    @classmethod
    def perspective_extract(cls,
                            image: np.ndarray,
                            out_width: int,
                            out_height: int,
                            yaw: float = 0.0,
                            pitch: float = 0.0,
                            roll: float = 0.0,
                            fov_degrees: float = 90.0,
                            interpolation: str = 'lanczos') -> np.ndarray:
        """Extract a rectilinear perspective view from an equirectangular image."""
        in_h, in_w = image.shape[:2]
        map_x, map_y = cls._cached_perspective_maps(in_w, in_h, out_width, out_height, yaw, pitch, roll, fov_degrees)

        # Prepare a dummy image to remap into desired output size by mapping grid shape
        # cv2.remap uses size of map to determine output size
        if interpolation == 'nearest':
            cv_interp = cv2.INTER_NEAREST
        elif interpolation == 'bilinear':
            cv_interp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            cv_interp = cv2.INTER_CUBIC
        elif interpolation == 'lanczos':
            cv_interp = cv2.INTER_LANCZOS4
        else:
            cv_interp = cv2.INTER_CUBIC

        out = cv2.remap(image, map_x, map_y, interpolation=cv_interp, borderMode=cv2.BORDER_WRAP)
        return out
    
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
    def _torch_rotation_matrix(yaw: float, pitch: float, roll: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.tensor(np.radians(yaw), device=device, dtype=dtype)
        p = torch.tensor(np.radians(pitch), device=device, dtype=dtype)
        r = torch.tensor(np.radians(roll), device=device, dtype=dtype)

        cy, sy = torch.cos(y), torch.sin(y)
        cp, sp = torch.cos(p), torch.sin(p)
        cr, sr = torch.cos(r), torch.sin(r)

        # ZYX order (yaw around Z, pitch around Y, roll around X)
        Rz = torch.stack([
            torch.stack([cy, -sy, torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([sy,  cy, torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype)])
        ])
        Ry = torch.stack([
            torch.stack([cp, torch.zeros((), device=device, dtype=dtype), sp]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([-sp, torch.zeros((), device=device, dtype=dtype), cp])
        ])
        Rx = torch.stack([
            torch.stack([torch.ones((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), cr, -sr]),
            torch.stack([torch.zeros((), device=device, dtype=dtype), sr,  cr])
        ])
        R = Rz @ Ry @ Rx
        return R

    @classmethod
    def torch_rotate_equirectangular(cls,
                                     image: torch.Tensor,
                                     yaw: float = 0.0,
                                     pitch: float = 0.0,
                                     roll: float = 0.0,
                                     horizon_offset: float = 0.0,
                                     interpolation: str = 'bilinear') -> torch.Tensor:
        """Rotate equirectangular image using torch.grid_sample.

        Args:
            image: Tensor (H, W, C) float32 on CPU/GPU in [0,1].
        Returns:
            Rotated image tensor (H, W, C) on same device.
        """
        device = image.device
        dtype = image.dtype
        H, W, C = image.shape

        # Create coordinate grid
        xs = torch.linspace(0.5, W - 0.5, W, device=device, dtype=dtype)
        ys = torch.linspace(0.5, H - 0.5, H, device=device, dtype=dtype)
        xg, yg = torch.meshgrid(xs, ys, indexing='xy')  # (W,H) -> we'll transpose later
        # Convert to spherical
        lon = (xg / W) * (2 * np.pi) - np.pi
        lat = (np.pi / 2) - (yg / H) * np.pi
        lat = lat + torch.tensor(np.radians(horizon_offset), device=device, dtype=dtype)
        lat = torch.clamp(lat, -np.pi/2, np.pi/2)

        # Spherical to Cartesian
        cos_lat = torch.cos(lat)
        x = cos_lat * torch.cos(lon)
        y = cos_lat * torch.sin(lon)
        z = torch.sin(lat)

        # Apply rotation
        R = cls._torch_rotation_matrix(yaw, pitch, roll, device, dtype)
        pts = torch.stack([x, y, z], dim=-1)  # (W,H,3)
        rot = torch.tensordot(pts, R.T, dims=1)  # (W,H,3)
        xr, yr, zr = rot[..., 0], rot[..., 1], rot[..., 2]

        # Back to spherical
        lat_r = torch.asin(torch.clamp(zr, -1.0, 1.0))
        lon_r = torch.atan2(yr, xr)

        # Back to pixel coordinates
        x_src = ((lon_r + np.pi) / (2 * np.pi)) * W
        y_src = ((np.pi/2 - lat_r) / np.pi) * H
        x_src = torch.remainder(x_src, W)
        y_src = torch.clamp(y_src, 0, H - 1)

        # grid_sample expects (N,C,H,W) and grid (N,H,W,2) in [-1,1]
        grid_x = (2.0 * x_src / (W - 1)) - 1.0
        grid_y = (2.0 * y_src / (H - 1)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

        inp = image.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        mode = 'bilinear' if interpolation in ('bilinear', 'lanczos', 'bicubic') else 'nearest'
        out = torch.nn.functional.grid_sample(inp, grid, mode=mode, padding_mode='border', align_corners=True)
        out = out.squeeze(0).permute(1, 2, 0)  # (H,W,C)
        return out

    @classmethod
    def torch_perspective_extract(cls,
                                  image: torch.Tensor,
                                  out_width: int,
                                  out_height: int,
                                  yaw: float = 0.0,
                                  pitch: float = 0.0,
                                  roll: float = 0.0,
                                  fov_degrees: float = 90.0,
                                  interpolation: str = 'bilinear') -> torch.Tensor:
        """Perspective extractor using torch.grid_sample.

        Args:
            image: (H,W,C) float in [0,1]
        Returns:
            (out_height, out_width, C)
        """
        device = image.device
        dtype = image.dtype
        H, W, C = image.shape

        xs = torch.linspace(0.5, out_width - 0.5, out_width, device=device, dtype=dtype)
        ys = torch.linspace(0.5, out_height - 0.5, out_height, device=device, dtype=dtype)
        u, v = torch.meshgrid(xs, ys, indexing='xy')

        f = out_width / (2.0 * np.tan(np.radians(fov_degrees) / 2.0))
        f = torch.tensor(f, device=device, dtype=dtype)
        cx = out_width / 2.0
        cy = out_height / 2.0

        x_cam = (u - cx) / f
        y_cam = (v - cy) / f
        z_cam = torch.ones_like(x_cam)
        norm = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
        x_cam /= norm; y_cam /= norm; z_cam /= norm

        R = cls._torch_rotation_matrix(yaw, pitch, roll, device, dtype)
        pts = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        rot = torch.tensordot(pts, R.T, dims=1)
        xr, yr, zr = rot[..., 0], rot[..., 1], rot[..., 2]
        lat = torch.asin(torch.clamp(zr, -1.0, 1.0))
        lon = torch.atan2(yr, xr)

        x_src = ((lon + np.pi) / (2 * np.pi)) * W
        y_src = ((np.pi/2 - lat) / np.pi) * H
        x_src = torch.remainder(x_src, W)
        y_src = torch.clamp(y_src, 0, H - 1)

        grid_x = (2.0 * x_src / (W - 1)) - 1.0
        grid_y = (2.0 * y_src / (H - 1)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        inp = image.permute(2, 0, 1).unsqueeze(0)
        mode = 'bilinear' if interpolation in ('bilinear', 'lanczos', 'bicubic') else 'nearest'
        out = torch.nn.functional.grid_sample(inp, grid, mode=mode, padding_mode='border', align_corners=True)
        out = out.squeeze(0).permute(1, 2, 0)
        return out

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
            face_img = cv2.remap(image, map_x, map_y, interpolation=cv_interp, borderMode=cv2.BORDER_WRAP)
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
    def cubemap_to_equirectangular(cls,
                                    cubemap_atlas: np.ndarray,
                                    output_width: int = 2048,
                                    output_height: int = 1024,
                                    layout: str = '3x2',
                                    interpolation: str = 'lanczos') -> np.ndarray:
        """Convert a 3x2 cubemap atlas back to equirectangular format.

        Args:
            cubemap_atlas: Input atlas with shape (2*face_size, 3*face_size, C).
            output_width: Width of output equirectangular image.
            output_height: Height of output equirectangular image.
            layout: Cubemap layout (currently only '3x2' supported).
            interpolation: Resampling method.

        Returns:
            Equirectangular image of shape (output_height, output_width, C).
        """
        if layout != '3x2':
            raise ValueError("Only '3x2' layout is currently supported")

        # Extract face size from atlas
        atlas_h, atlas_w = cubemap_atlas.shape[:2]
        face_size = atlas_h // 2

        # Extract individual faces from atlas
        # Layout: [left, front, right] / [back, top, bottom]
        faces = {
            'left': cubemap_atlas[0:face_size, 0:face_size],
            'front': cubemap_atlas[0:face_size, face_size:2*face_size],
            'right': cubemap_atlas[0:face_size, 2*face_size:3*face_size],
            'back': cubemap_atlas[face_size:2*face_size, 0:face_size],
            'top': cubemap_atlas[face_size:2*face_size, face_size:2*face_size],
            'bottom': cubemap_atlas[face_size:2*face_size, 2*face_size:3*face_size],
        }

        # Create output coordinate grid
        xs = np.linspace(0.5, output_width - 0.5, output_width, dtype=np.float32)
        ys = np.linspace(0.5, output_height - 0.5, output_height, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(xs, ys)

        # Convert to spherical coordinates
        lat, lon = cls.equirectangular_to_spherical(x_grid, y_grid, output_width, output_height)

        # Convert to 3D Cartesian
        x, y, z = cls.spherical_to_cartesian(lat, lon)

        # Determine which face each pixel should sample from
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)

        # Initialize output
        output = np.zeros((output_height, output_width, cubemap_atlas.shape[2]), dtype=cubemap_atlas.dtype)

        def sample_face(face_name: str, mask: np.ndarray):
            """Sample from a specific cube face for masked pixels."""
            if not np.any(mask):
                return

            face_img = faces[face_name]
            x_m, y_m, z_m = x[mask], y[mask], z[mask]

            # Map 3D direction to face UV coordinates
            if face_name == 'front':  # +X
                u = y_m / x_m
                v = z_m / x_m
            elif face_name == 'back':  # -X
                u = -y_m / -x_m
                v = z_m / -x_m
            elif face_name == 'right':  # +Y
                u = -x_m / y_m
                v = z_m / y_m
            elif face_name == 'left':  # -Y
                u = x_m / -y_m
                v = z_m / -y_m
            elif face_name == 'top':  # +Z
                u = x_m / z_m
                v = y_m / z_m
            elif face_name == 'bottom':  # -Z
                u = x_m / -z_m
                v = -y_m / -z_m

            # Convert UV from [-1, 1] to pixel coordinates [0, face_size-1]
            u_px = (u + 1.0) * 0.5 * (face_size - 1)
            v_px = (v + 1.0) * 0.5 * (face_size - 1)

            # Clamp to valid range
            u_px = np.clip(u_px, 0, face_size - 1)
            v_px = np.clip(v_px, 0, face_size - 1)

            # Interpolate using cv2.remap
            if interpolation == 'nearest':
                cv_interp = cv2.INTER_NEAREST
            elif interpolation == 'bilinear':
                cv_interp = cv2.INTER_LINEAR
            elif interpolation == 'bicubic':
                cv_interp = cv2.INTER_CUBIC
            else:
                cv_interp = cv2.INTER_LANCZOS4

            # Create a dense grid for the masked region
            mask_indices = np.where(mask)
            for i, (row, col) in enumerate(zip(mask_indices[0], mask_indices[1])):
                # Sample individual pixel
                x_sample = u_px[i]
                y_sample = v_px[i]

                if interpolation == 'nearest':
                    x_int = int(np.round(x_sample))
                    y_int = int(np.round(y_sample))
                    output[row, col] = face_img[y_int, x_int]
                else:
                    # Bilinear interpolation
                    x0 = int(np.floor(x_sample))
                    x1 = min(x0 + 1, face_size - 1)
                    y0 = int(np.floor(y_sample))
                    y1 = min(y0 + 1, face_size - 1)

                    wx1 = x_sample - x0
                    wx0 = 1.0 - wx1
                    wy1 = y_sample - y0
                    wy0 = 1.0 - wy1

                    output[row, col] = (
                        face_img[y0, x0] * wx0 * wy0 +
                        face_img[y0, x1] * wx1 * wy0 +
                        face_img[y1, x0] * wx0 * wy1 +
                        face_img[y1, x1] * wx1 * wy1
                    )

        # Determine face selection for each pixel
        front_mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
        back_mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
        right_mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
        left_mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y < 0)
        top_mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
        bottom_mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z < 0)

        # Sample from each face
        sample_face('front', front_mask)
        sample_face('back', back_mask)
        sample_face('right', right_mask)
        sample_face('left', left_mask)
        sample_face('top', top_mask)
        sample_face('bottom', bottom_mask)

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
