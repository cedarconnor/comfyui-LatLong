# ComfyUI LatLong — Equirectangular Image Nodes

High‑quality equirectangular (360°) image processing nodes for ComfyUI. Rotate with yaw/pitch/roll, adjust the horizon, crop to 180° or square, convert to cubemaps, and extract perspective views — all designed for panoramic workflows.

## What's New

- **Five new nodes added**: Cubemap ↔ Equirectangular conversion, Mirror/Flip, Resize with aspect preservation, Preset rotations, and Cubemap face extraction
- All‑in‑One node now embeds controls directly in the node (sliders/toggles/dropdowns). No external parameter nodes are required for rotation/crop/interpolation.
- Helpful tooltips added to every node and parameter to clarify behavior and best‑use guidance.

## Installation

Option A — Clone (recommended)
- From `ComfyUI/custom_nodes`:
  - `git clone https://github.com/cedarconnor/comfyui-LatLong.git`
  - `cd comfyui-LatLong`
  - `pip install -r requirements.txt`
  - Restart ComfyUI

Option B — Manual
- Extract the repo into `ComfyUI/custom_nodes/comfyui-LatLong/`
- Install deps: `pip install numpy>=1.21.0 opencv-python>=4.5.0 scipy>=1.7.0 torch>=1.9.0`
- Restart ComfyUI

Requirements
- ComfyUI (latest), Python 3.8+
- Python packages:
  - numpy>=1.21.0, opencv-python>=4.5.0, scipy>=1.7.0, torch>=1.9.0

GPU Notes
- Some GPU paths use `torch.grid_sample` and support bilinear/nearest only. Lanczos/Bicubic use CPU (OpenCV) paths.

## Nodes Overview

### Core Transformation Nodes

1) **Equirectangular Rotate**
- Purpose: Rotate an equirectangular image with yaw/pitch/roll and adjust the horizon.
- Inputs:
  - `image`: equirectangular tensor (B,H,W,C) in [0,1]
  - `yaw_rotation`: horizontal rotation (degrees)
  - `pitch_rotation`: vertical tilt (degrees)
  - `roll_rotation`: bank rotation (degrees)
  - `horizon_offset`: vertical horizon shift (degrees)
  - `interpolation`: lanczos | bicubic | bilinear | nearest
  - `backend`: auto | cpu | gpu

2) **Equirectangular Rotate (Preset)**
- Purpose: Quick rotation to preset views with fine-tuning options.
- Presets: front, back, left, right, up, down
- Additional controls: yaw/pitch/roll offsets, horizon adjustment
- Ideal for: Quickly orienting panoramas to standard viewing angles

3) **Equirectangular Crop 180**
- Purpose: Extract a horizontal FOV (default 180°) window from an equirectangular image.
- Inputs:
  - `image`
  - `output_width`, `output_height`: target size
  - `maintain_aspect`: preserve natural aspect ratio for selected FOV
  - `center_longitude_deg`: FOV center longitude (degrees)
  - `fov_degrees`: horizontal field of view (degrees)
  - `interpolation`

4) **Equirectangular Crop Square**
- Purpose: Center‑crop width to match the original height (perfect square).
- Inputs: `image`, `interpolation` (not used by pure crop; kept for consistency)

5) **Equirectangular Processor (All‑in‑One)**
- Purpose: Rotate then optionally crop to 180° or to a centered square — a single‑node workflow. Note: square crop takes precedence over 180°.
- Inputs:
  - `image`
  - `yaw_rotation`, `pitch_rotation`, `roll_rotation`, `horizon_offset`
  - `crop_to_180`, `crop_to_square`
  - `output_width`, `output_height` (for 180° crop)
  - `interpolation`

6) **Equirectangular Perspective Extract**
- Purpose: Extract a pinhole‑camera perspective view (yaw/pitch/roll/FOV) from an equirectangular panorama.
- Inputs: `image`, `yaw_rotation`, `pitch_rotation`, `roll_rotation`, `fov_degrees`, `output_width`, `output_height`, `interpolation`, `backend`

### Cubemap Conversion Nodes

7) **Equirectangular → Cubemap (3×2)**
- Purpose: Convert equirectangular panorama to a 3×2 cubemap atlas.
- Inputs: `image`, `face_size`, `layout` (3×2), `interpolation`
- Face order: [left, front, right] on top row; [back, top, bottom] on bottom row

8) **Cubemap (3×2) → Equirectangular**
- Purpose: Convert a 3×2 cubemap atlas back to equirectangular format.
- Inputs: `cubemap_atlas`, `output_width`, `output_height`, `layout`, `interpolation`
- Useful for: Round-trip workflows, importing cubemaps from other tools

9) **Cubemap Faces Extract**
- Purpose: Extract individual cube faces from equirectangular panorama as separate images.
- Outputs: 6 separate images (left, front, right, back, top, bottom)
- Inputs: `image`, `face_size`, `interpolation`
- Useful for: Processing individual faces, creating custom cubemap layouts

### Utility Nodes

10) **Equirectangular Mirror/Flip**
- Purpose: Mirror or flip panoramas with proper spherical wrapping.
- Options:
  - `mirror_horizontal`: flip left-right (reverse longitude)
  - `mirror_vertical`: flip top-bottom (reverse latitude)
- Useful for: Correcting orientation, creating mirrored effects

11) **Equirectangular Resize**
- Purpose: Resize equirectangular images with aspect ratio preservation.
- Inputs: `image`, `output_width`, `output_height`, `maintain_aspect`, `interpolation`
- Features: Auto-maintains 2:1 aspect ratio (standard equirectangular) or custom dimensions

## Quickstart Examples

**Basic Rotation**
- Add Equirectangular Rotate, connect an image, tweak yaw/pitch/roll/horizon.

**Quick Preset Views**
- Add Equirectangular Rotate (Preset), select preset (front/back/left/right/up/down), optionally fine-tune with offsets.

**Square Crop**
- Add Equirectangular Crop Square to center‑crop to a perfect square.

**180° Extraction**
- Add Equirectangular Crop 180, set `center_longitude_deg` and `fov_degrees` as needed, enable Maintain Aspect for correct proportions.

**Perspective View**
- Add Equirectangular Perspective Extract, set yaw/pitch/roll, FOV, and output size.

**Cubemap Generation**
- Add Equirectangular → Cubemap (3×2), choose `face_size`, select interpolation.

**Cubemap Conversion**
- Generate cubemap with Equirectangular → Cubemap, then convert back with Cubemap → Equirectangular for round-trip workflows.

**Extract Individual Faces**
- Add Cubemap Faces Extract to get 6 separate images (left, front, right, back, top, bottom) from an equirectangular panorama.

**Mirror/Flip**
- Add Equirectangular Mirror/Flip, enable horizontal or vertical flip for orientation correction or creative effects.

**Smart Resize**
- Add Equirectangular Resize with maintain_aspect=True to automatically preserve 2:1 ratio, or disable for custom dimensions.

**All‑in‑One**
- Add Equirectangular Processor (All‑in‑One), adjust rotation; enable square or 180° crop; pick interpolation. All controls are embedded in the node.

## Technical Notes

Coordinate System
- Accurate spherical transforms with longitude wrapping and pole handling.

Interpolation
- CPU: OpenCV remap (Lanczos, Bicubic, Bilinear, Nearest)
- GPU: torch.grid_sample (Bilinear/Nearest where applicable)

Pipeline
- Input (B,H,W,C) → float32 processing → transform → resample → output tensor in [0,1].

## Troubleshooting

- Blurry output: prefer Lanczos/Bicubic for final renders.
- Memory limits: reduce resolution or batch size.
- GPU path missing filters: use CPU for Lanczos/Bicubic.

## Contributing

Issues, feature requests, and PRs are welcome.

## License

Open source — see the repository license.

— Ideal for 360° photography, VR, panoramic pipelines, and social content where precise equirectangular control matters.
