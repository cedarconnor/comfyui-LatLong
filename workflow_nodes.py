"""Panorama editing workflows built on the same geometry as the original nodes."""
import json
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation, Slerp
from .modules.equirectangular_processor import EquirectangularProcessor as P
from .nodes import (_validate_image, _perspective_insert, PanoramaViewerNode, PanoramaVideoViewerNode,
                    EquirectangularToCubemapFlexible, CubemapToEquirectangularFlexible,
                    _parse_face_order, _atlas_3x2_from_faces, _dice_from_faces, _horizon_from_faces,
                    _faces_from_atlas_3x2, _faces_from_dice, _faces_from_horizon)


class LatLongFloatProcessor:
    CATEGORY = "LatLong/HDR"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    DESCRIPTION = "Preserve finite scene-linear values, including negatives and values above one. Preview tone mapping is separate."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "operation": (["rotate", "perspective", "to_cubemap", "from_cubemap", "resize"],),
            "yaw": ("FLOAT", {"default": 0.}), "pitch": ("FLOAT", {"default": 0.}), "roll": ("FLOAT", {"default": 0.}),
            "fov": ("FLOAT", {"default": 90., "min": 1., "max": 179.}),
            "width": ("INT", {"default": 2048, "min": 1, "max": 32768}), "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
            "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"],)}}

    def process(self, image, operation, yaw, pitch, roll, fov, width, height, interpolation):
        _validate_image(image)
        results = []
        for frame in image:
            a = frame.cpu().numpy().astype(np.float32, copy=False)
            if operation == "rotate": result = P.rotate_equirectangular(a, yaw, pitch, roll, interpolation=interpolation, use_tiling=True)
            elif operation == "perspective": result = P.perspective_extract(a, width, height, yaw, pitch, roll, fov, interpolation)
            elif operation == "to_cubemap": result = P.equirectangular_to_cubemap(a, width, interpolation=interpolation)
            elif operation == "from_cubemap": result = P.cubemap_to_equirectangular(a, width, height, interpolation=interpolation)
            elif operation == "resize": result = P.resize_equirectangular(a, width, height, False, interpolation)
            else: raise ValueError("Unknown operation")
            if result.ndim == 2: result = result[..., None]
            results.append(torch.from_numpy(result))
        return (torch.stack(results),)


class LatLongToneMap:
    CATEGORY = "LatLong/HDR"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "exposure_stops": ("FLOAT", {"default": 0., "min": -20., "max": 20.})}}

    def process(self, image, exposure_stops):
        _validate_image(image)
        x = image.clamp(min=0) * (2 ** exposure_stops)
        x = x / (1 + x)
        return (torch.where(x <= .0031308, 12.92 * x, 1.055 * x.pow(1 / 2.4) - .055),)


class LatLongExtractProjection:
    CATEGORY = "LatLong/Outpaint"
    RETURN_TYPES = ("IMAGE", "MASK", "PROJECTION_CONTEXT")
    RETURN_NAMES = ("patch", "coverage", "projection_context")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "yaw": ("FLOAT", {"default": 0.}), "pitch": ("FLOAT", {"default": 0.}),
            "roll": ("FLOAT", {"default": 0.}), "fov": ("FLOAT", {"default": 90., "min": 1., "max": 179.}),
            "width": ("INT", {"default": 1024, "min": 1, "max": 8192}), "height": ("INT", {"default": 1024, "min": 1, "max": 8192})}}

    def extract(self, image, yaw, pitch, roll, fov, width, height):
        _validate_image(image)
        h, w = image.shape[1:3]
        context = dict(version=1, yaw=yaw, pitch=pitch, roll=roll, fov=fov,
                       width=width, height=height, panorama_width=w, panorama_height=h,
                       basis="X-front Y-right Z-up; extrinsic zyx; integer equirectangular centers")
        maps = P._cached_perspective_maps(w, h, width, height, yaw, pitch, roll, fov)
        patches = [torch.from_numpy(P.interpolate_image(frame.cpu().numpy(), *maps, method="bilinear")) for frame in image]
        _, mask = _perspective_insert(torch.ones((1, height, width), device=image.device), h, w, yaw, pitch, roll, fov)
        return torch.stack(patches), mask.repeat(len(image), 1, 1), context


class LatLongReinsertProjection:
    CATEGORY = "LatLong/Outpaint"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"panorama": ("IMAGE",), "edited_patch": ("IMAGE",), "projection_context": ("PROJECTION_CONTEXT",),
                             "feather": ("INT", {"default": 16, "min": 0, "max": 512})}}

    def stitch(self, panorama, edited_patch, projection_context, feather):
        _validate_image(panorama); _validate_image(edited_patch)
        c = projection_context
        if c.get("version") != 1: raise ValueError("Unsupported projection context version")
        if edited_patch.shape[2] * c["height"] != edited_patch.shape[1] * c["width"]:
            raise ValueError("Edited patch aspect ratio must match the extracted projection")
        if panorama.shape[-1] != edited_patch.shape[-1]: raise ValueError("Patch and panorama channel counts must match")
        outputs = []
        for i, background in enumerate(panorama):
            patch = edited_patch[i % len(edited_patch)].to(background.device).permute(2, 0, 1)
            layer, alpha = _perspective_insert(patch, *background.shape[:2], c["yaw"], c["pitch"], c["roll"], c["fov"], feather)
            outputs.append((alpha * layer + (1 - alpha) * background.permute(2, 0, 1)).permute(1, 2, 0))
        return (torch.stack(outputs),)


class LatLongDiagnostics:
    CATEGORY = "LatLong/Analysis"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("diagnostic_overlay", "measurements")
    FUNCTION = "analyze"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "band_width": ("INT", {"default": 10, "min": 1, "max": 4096})}}

    def analyze(self, image, band_width):
        _validate_image(image)
        reports, overlays = [], []
        for frame in image:
            h, w, c = frame.shape; band = min(band_width, max(1, w // 2 - 1))
            seam = (frame[:, 0] - frame[:, -1]).abs()
            reports.append(dict(wrap_mean=float(seam.mean()), wrap_max=float(seam.max()),
                left_join=float((frame[:, band] - frame[:, band-1]).abs().mean()) if w > 1 else 0.,
                right_join=float((frame[:, -band] - frame[:, -band-1]).abs().mean()) if w > 1 else 0.,
                north_variation=float(frame[0].var(dim=0, unbiased=False).mean()),
                south_variation=float(frame[-1].var(dim=0, unbiased=False).mean())))
            overlay = frame[..., :3].clamp(0, 1).clone()
            if c == 1: overlay = overlay.repeat(1, 1, 3)
            overlay[:, [0, w-1]] = torch.tensor([1., 0., 0.], device=frame.device)
            if w > 1: overlay[:, [band, w-band-1]] = torch.tensor([1., .7, 0.], device=frame.device)
            overlay[[0, h-1], :] = torch.tensor([0., .7, 1.], device=frame.device)
            overlays.append(overlay)
        return torch.stack(overlays), json.dumps(reports, indent=2)


class LatLongComparePanorama:
    CATEGORY = "LatLong"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "compare"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"before": ("IMAGE",), "after": ("IMAGE",), "max_width": ("INT", {"default": 2048, "min": 64, "max": 8192})},
                "optional": {"fps": ("INT", {"default": 24, "min": 1, "max": 120})}}

    def compare(self, before, after, max_width, fps=24):
        _validate_image(before); _validate_image(after)
        if len(before) != len(after) and min(len(before), len(after)) != 1:
            raise ValueError("Comparison batches must match, or one side must contain one image")
        if max(len(before), len(after)) > 1:
            viewer = PanoramaVideoViewerNode()
            a = viewer.view_video_pano(before, fps, max_width)["ui"]["pano_video_frames"]
            b = viewer.view_video_pano(after, fps, max_width)["ui"]["pano_video_frames"]
            count = max(len(a), len(b))
            return {"ui": {"pano_video_frames": a * count if len(a) == 1 else a,
                           "compare_frames": b * count if len(b) == 1 else b, "fps": [str(fps)]}}
        viewer = PanoramaViewerNode()
        return {"ui": {"pano_image": viewer.view_pano(before, max_width)["ui"]["pano_image"],
                       "compare_image": viewer.view_pano(after, max_width)["ui"]["pano_image"]}}


class LatLongAnimatedRotation:
    CATEGORY = "LatLong/Animation"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "keyframes": ("STRING", {"multiline": True,
            "default": '[{"frame":0,"yaw":0,"pitch":0,"roll":0},{"frame":29,"yaw":90,"pitch":0,"roll":0}]'}),
            "frame_count": ("INT", {"default": 30, "min": 1, "max": 4096}),
            "interpolation": (["bilinear", "lanczos", "bicubic", "nearest"],)}}

    def rotate(self, image, keyframes, frame_count, interpolation):
        _validate_image(image)
        keys = json.loads(keyframes)
        times = np.array([k["frame"] for k in keys], dtype=float)
        angles = np.array([[k.get(a, 0.) for a in ("yaw", "pitch", "roll")] for k in keys])
        if len(keys) < 2 or not np.isfinite(times).all() or not np.isfinite(angles).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Provide at least two finite keyframes in increasing frame order")
        rotations = Slerp(times, Rotation.from_euler("zyx", angles, degrees=True))(np.clip(np.arange(frame_count), times[0], times[-1]))
        outputs = []
        for i, angles in enumerate(rotations.as_euler("zyx", degrees=True)):
            source = image[i % len(image)].cpu().numpy()
            outputs.append(torch.from_numpy(P.rotate_equirectangular(source, *angles, interpolation=interpolation, use_tiling=True)))
        return (torch.stack(outputs),)


class LatLongCubemapPreset:
    CATEGORY = "LatLong/Cubemap"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    PRESETS = {"LatLong 3x2 (L F R / B U D)": ("atlas_3x2", "L,F,R,B,U,D"),
               "Cross 4x3 (U / L F R B / D)": ("dice", "F,R,B,L,U,D"),
               "Strip FRBLUD": ("horizon", "F,R,B,L,U,D"),
               "Stack FRBLUD": ("stack", "F,R,B,L,U,D")}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "direction": (["export", "import"],), "preset": (list(cls.PRESETS),),
            "face_size": ("INT", {"default": 512, "min": 16, "max": 4096}), "panorama_width": ("INT", {"default": 2048, "min": 32, "max": 16384}),
            "flip_face_vertical": ("BOOLEAN", {"default": False})}}

    def convert(self, image, direction, preset, face_size, panorama_width, flip_face_vertical):
        _validate_image(image)
        fmt, order = self.PRESETS[preset]
        if direction == "export":
            # Vertical flipping is explicit; no undocumented vendor coordinate assumptions.
            if flip_face_vertical:
                stack = EquirectangularToCubemapFlexible().to_cubemap_flexible(image, face_size, "stack", order, "bilinear")[0].flip(1)
                return (_pack_stack(stack, fmt, order),)
            return EquirectangularToCubemapFlexible().to_cubemap_flexible(image, face_size, fmt, order, "bilinear")
        if flip_face_vertical:
            image = _unpack_layout(image, fmt, order).flip(1); fmt = "stack"
        return CubemapToEquirectangularFlexible().to_equirectangular_flexible(image, fmt, order, panorama_width, panorama_width // 2, "bilinear")


def _pack_stack(stack, fmt, order):
    if fmt == "stack": return stack
    names = _parse_face_order(order); outputs = []
    for start in range(0, len(stack), 6):
        faces = {name: stack[start+i].cpu().numpy() for i, name in enumerate(names)}
        array = _atlas_3x2_from_faces(faces) if fmt == "atlas_3x2" else _dice_from_faces(faces) if fmt == "dice" else _horizon_from_faces(faces, names)
        outputs.append(torch.from_numpy(array))
    return torch.stack(outputs)


def _unpack_layout(image, fmt, order):
    if fmt == "stack": return image
    names = _parse_face_order(order); outputs = []
    for frame in image:
        a = frame.cpu().numpy()
        if fmt == "atlas_3x2": faces = _faces_from_atlas_3x2(a, a.shape[0] // 2)
        elif fmt == "dice": faces = _faces_from_dice(a, a.shape[0] // 3)
        else: faces = _faces_from_horizon(a, names)
        outputs.extend(torch.from_numpy(faces[name]) for name in names)
    return torch.stack(outputs)


WORKFLOW_NODES = {"LatLong Float Processor": LatLongFloatProcessor, "LatLong Tone Map Preview": LatLongToneMap,
    "LatLong Extract Projection": LatLongExtractProjection, "LatLong Reinsert Projection": LatLongReinsertProjection,
    "LatLong Diagnostics": LatLongDiagnostics, "LatLong Compare Panorama": LatLongComparePanorama,
    "LatLong Animated Rotation": LatLongAnimatedRotation, "LatLong Cubemap Preset": LatLongCubemapPreset}
