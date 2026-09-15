# ComfyUI LatLong

Equirectangular image transforms, cubemap conversion, perspective editing, outpainting, and interactive panorama previews for ComfyUI.

## Install

Clone into `ComfyUI/custom_nodes/comfyui-LatLong`, install `requirements.txt` with ComfyUI's Python, and restart ComfyUI. Python 3.10 or later is required. Three.js r172 is bundled, so previewing does not require a CDN connection.

```powershell
git clone https://github.com/cedarconnor/comfyui-LatLong.git
cd comfyui-LatLong
python -m pip install -r requirements.txt
```

For portable ComfyUI, replace `python` with its embedded Python executable. A Python wheel is also buildable; normal ComfyUI discovery still requires installing the extension in `custom_nodes`.

## Changes in 2.1

- CPU and GPU now use the same mixed-angle rotation convention. Neutral and integer-pixel yaw rotations preserve pixels.
- Perspective extraction faces the equator at zero angles and shares the insertion camera basis.
- Sampling wraps longitude without wrapping the north and south poles together.
- Bicubic/Lanczos use CPU; requesting them explicitly with GPU produces a clear error. Auto considers filter support and available VRAM.
- Rotation uses bounded working bands; CPU source tiles also avoid OpenCV's 32767-pixel source limit.
- Cubemap reconstruction is vectorized and honors all four interpolation modes.
- Edge blending tapers to the untouched interior and matches both wrap edges.
- Outpaint perspective scale changes angular coverage. Stitch supports alpha, overlay, and hard blending. 2D patches wrap horizontally.
- Viewers release textures and WebGL resources, discard stale loads, pause rendering while hidden, and provide play/pause/scrubbing.
- Eight workflow nodes add float processing, preview tone mapping, reusable projections, diagnostics, comparison, animation, and cubemap presets.

### Existing workflow changes

Node names and output sockets are retained. Corrected GPU mixed-angle rotations and perspective extraction can change old results. CPU extrinsic `zyx` rotation order remains the reference. Perspective scale changes projected FOV rather than resampling source resolution. The `overlay` stitch option now performs the standard overlay blend; `hard` disables feathering.

Circular-padding nodes now register changes through ComfyUI's model patcher. `inplace=False` clones the patcher instead of duplicating all model weights. `inplace=True` modifies the supplied patcher. This affects Conv2d layers only; it does not make transformer models, all VAE architectures, or every downstream operation seam-free.

## Geometry contract

- World +X: front / longitude 0; +Y: longitude +90; +Z: north.
- Angles are degrees. Rotation uses SciPy extrinsic `zyx`: yaw around Z, pitch around Y, roll around X. Positive pitch follows that right-handed rotation and points the forward direction downward.
- An equirectangular pixel `(x,y)` maps to longitude `360*x/W-180` and latitude `90-180*y/H`. Integer coordinates are pixel centers.
- A perspective camera faces world +X at zero angles, with image-right +Y and image-down -Z. FOV is horizontal; pixels are square.
- Longitude is periodic; latitude clamps at the first/last row. Cubemap faces clamp at their own borders during reconstruction; no cross-face high-order filtering is claimed.
- Legacy 3x2 atlas: `[left, front, right] / [back, top, bottom]`. Legacy face raster v increases toward its defined positive face-v direction. Presets retain this convention and offer an explicit reversible vertical flip.
- Original transform nodes return float32 IMAGE values clipped to [0,1]. Use **LatLong Float Processor** for scene-linear/HDR data. Masks are BHW, with 1 meaning selected/inpaint unless an output is explicitly called coverage.

## Nodes

### Transforms and cubemaps

| Node | Behavior |
|---|---|
| Equirectangular Rotate | Yaw/pitch/roll and horizon offset, CPU/GPU and tiling controls |
| Equirectangular Rotate (Preset) | Front/back/left/right/up/down with offsets |
| Equirectangular Processor (All-in-One) | Rotate and optionally crop; square takes precedence |
| Equirectangular Crop 180 | Seam-safe longitude window with adjustable FOV and center |
| Equirectangular Crop Square | Center width crop to input height |
| Equirectangular Perspective Extract | Rectilinear camera view |
| Equirectangular Mirror/Flip | Horizontal and/or vertical image flip |
| Equirectangular Resize | Optional 2:1 output aspect |
| Equirectangular to/from Cubemap | Legacy 3x2 atlas |
| Cubemap Faces Extract | Left, front, right, back, top, bottom outputs |
| Flexible cubemap converters | 3x2, cross/dice, horizontal strip, or B*6 face stack |
| Stack / Split Cubemap Faces | Reversible face stack with explicit face order |
| LatLong Cubemap Preset | Export/import known layouts with a reversible face vertical flip |

Flexible `list`/`dict` choices remain aliases of stack for old workflows. Face names accept F/R/B/L/U/D and full names. Presets describe layouts, not undocumented Unreal/Unity/OpenGL axis conversions.

### Masks and seam editing

- **Create Seam Mask:** centered strip with optional feather and half-width roll.
- **Create Pole Mask:** circular face masks or north/south coverage in equirectangular space.
- **Roll Image / Roll Mask:** wrap without resampling.
- **Apply Circular Padding Model / VAE:** patch Conv2d padding in x only or both axes.
- **Equirectangular Edge Blender:** match the wrap boundary while tapering corrections into both edge bands.
- **LatLong Diagnostics:** RGB overlay plus JSON measurements for wrap mismatch, both selected band joins, and polar row variation. Red marks wrap edges, amber band joins, cyan poles. Metrics locate potential defects; they do not establish visual quality.

### Outpaint and projection editing

**Outpaint Setup → inpaint → Outpaint Stitch**

Setup returns a placed image, outpaint mask (1=inpaint), and stitch context. In 2D mode, scale is a source-pixel multiplier and x placement wraps across ±180°. Y is clipped. At sizes wider than the panorama, the first wrapped source span is sampled once. In perspective mode, scale multiplies `tan(FOV/2)`; source pixels are not enlarged. Increasing angular coverage changes perspective geometry. Feather is measured in source pixels for perspective placement.

Stitch regenerates the layer at the output resolution. Alpha blends normally, overlay uses the overlay color formula under the mask, and hard pastes without feathering. Contexts are versioned; original contexts without a version are accepted as version 1.

**LatLong Extract Projection → edit patch → LatLong Reinsert Projection**

Extraction returns an image patch, full-panorama coverage mask, and versioned projection context. Reinsertion uses the saved yaw/pitch/roll/FOV and source aspect. Upscaling the patch is supported if its aspect ratio is unchanged. Contexts contain geometry metadata rather than a hidden copy of the panorama. Values are not clipped in this path.

### HDR, animation, and preview

- **LatLong Float Processor:** rotate, perspective, cubemap conversion, or resize without clipping negative or above-one values. For `to_cubemap`, width is face size; for rotate, source dimensions are retained. CPU filters preserve floating values; export encoding is handled downstream.
- **LatLong Tone Map Preview:** exposure, Reinhard tone mapping, then linear-to-sRGB display conversion. Connect this to preview nodes while preserving the original HDR branch for output.
- **LatLong Animated Rotation:** JSON keyframes with frame/yaw/pitch/roll. Uses shortest-path quaternion interpolation, clamps outside the keyframe time range, and cycles the input batch when producing more frames than inputs. Example: `[{"frame":0,"yaw":0},{"frame":29,"yaw":90}]`.
- **Preview 360 Panorama:** first image of a batch, including grayscale; drag to look, wheel to zoom.
- **Preview 360 Video Panorama:** decoded textures are limited to the displayed frame; play/pause and an exact frame-index slider. Playback waits for decoding and can run below the requested FPS. Encoded frame URLs remain in the UI payload; very long/high-resolution batches can still use substantial host memory.
- **LatLong Compare Panorama:** before/after images or matching batches, shared camera, frame index, and wipe slider. Both images finish loading before the displayed comparison frame changes. A single image on either side is held across the other sequence.

## Memory and performance

Tiling bounds geometry/source working buffers, not the full IMAGE tensors. A 16384×8192 RGB float32 image is 1.5 GiB; input plus output alone require 3 GiB. Node batch stacking and upstream/downstream tensors add memory. There is no persistent map cache.

CPU rotation automatically tiles above one megapixel or an 8192-pixel dimension. `tile_size` is an upper bound; working bands are further capped to approximately 262144 pixels. Source tiles include the interpolation halo. Independent inverse samples need no overlap blending. GPU rotation also uses bounded bands; disabling tiling permits a full-height grid. Auto falls back to CPU when the estimated image working set exceeds its available-VRAM budget.

Run `python tests/benchmark.py 2048 4096 16384` for elapsed time and sampled process peak RSS on your hardware (`psutil` is a benchmark-only dependency). See `VALIDATION.md` for measured results and limits. No universal 16K/32K timing or sub-gigabyte memory guarantee is made.

## Development

```text
python -m unittest discover -s tests -p "test_*.py"
python tests/run_node_smoke_tests.py
npm test
python -m pip wheel --no-deps --no-build-isolation --wheel-dir output/dist .
```

CPU tests run in CI on Python 3.10 and 3.12. CUDA checks run when available. Publishing depends on tests passing and requires the actual Comfy Registry publisher ID plus `REGISTRY_ACCESS_TOKEN`; the publisher ID must be supplied by the maintainer.

Regression tests cover geometry, filters, large-source sampling, outpainting, HDR, projection context, animation, layouts, and viewer lifecycle. Tests do not qualify every diffusion model or VAE.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Bundled Three.js retains its [MIT license](js/lib/THREE-LICENSE.txt).
