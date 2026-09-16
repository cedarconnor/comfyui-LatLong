# LatLong 2.1 validation

Tested on Windows 11, NVIDIA RTX A6000 (48 GiB VRAM), approximately 256 GiB system RAM.

## Automated checks

- 24 Python regression tests pass on the portable ComfyUI runtime (Torch 2.12.0+cu130, NumPy 2.4.6, OpenCV 4.13.0), including CPU/CUDA geometry parity.
- The same 24 tests pass with Python 3.11.14, Torch 2.0.0+cpu, NumPy 1.24.0, SciPy 1.10.0, OpenCV 4.8.0.74, Pillow 9.5.0.
- The original smoke suite passes in both environments.
- Five JavaScript lifecycle tests pass: stale loads, decode locking, queued seeking, disposal, comparison partial failure, and single-frame indexing.
- A wheel builds successfully. An isolated installation imports all 34 nodes and contains the viewer, Three.js core/module, and third-party license.
- Real ComfyUI ModelPatcher tests pass with ordinary Conv2d and ComfyUI manual-cast Conv2d: clone isolation, applying padding, changing axes, restoring the original behavior, and copying the VAE patcher.

## Live ComfyUI and browser

A temporary ComfyUI instance on localhost:8191 loaded this extension with other custom nodes disabled. A real prompt completed GPU rotation, extraction/reinsertion, HDR preview, diagnostics, cubemap presets, animated preview, comparison, and perspective/2D outpaint stitching.

Browser checks with the installed frontend:

- Both comparison textures load and render under the same camera.
- Twenty comparison reloads retain a steady four renderer textures (source textures plus Three.js background textures).
- Video playback advances; pause and seeking work. End seeks to frame 4/4, with two renderer textures.
- The final live prompt completed with four synchronized before/after frames. The browser rendered both sequences with a shared frame index and four renderer textures.
- Browser screenshots are in `output/playwright/` (local generated evidence, not shipped).
- Startup logged missing optional user CSS/templates in the isolated user directory and a frontend graph-initialization warning; no LatLong module/texture errors were observed.

The first isolated startup triggered ComfyUI's legacy-database migration despite a separate user directory. The original legacy database was restored from the backup created by ComfyUI. Subsequent validation explicitly used an in-memory database.

## Memory and timing

Single RGB float32 image, yaw=23Â°, pitch=12Â°, roll=5Â°. CPU uses Lanczos; GPU uses bilinear, so these timings are not filter-equivalent comparisons. Source is constant gray to make output integrity independently checkable. Geometry still evaluates the full nontrivial rotation.

| Resolution | CPU seconds | CPU process peak RSS | GPU seconds | GPU peak allocated memory |
|---|---:|---:|---:|---:|
| 4096Ã—2048 | 2.81 | 0.77 GiB | 0.169 | 0.383 GiB |
| 16384Ã—8192 | 74.51 | 3.59 GiB | 0.624 | 6.008 GiB |

CPU RSS includes the Python/runtime baseline and is sampled every 10 ms. GPU timing excludes initial source upload/allocation and synchronizes around processing. GPU peak is PyTorch allocated memory, not total device usage. These are processor-level measurements, not full ComfyUI graph timings or batch-memory guarantees. GPU testing uses bounded 262144-pixel working bands.

Reducing the CPU source tile from 1024 to 256 improved the measured 16K run from 183.98 s to 74.51 s. The CPU input/output tensors alone occupy 3 GiB at 16K. A slim 32768-pixel-wide source is covered by a regression test; a full 32K panorama was not benchmarked.

## Remaining boundaries

- CI is configured but has not been run on GitHub in this task.
- Registry publication requires the maintainer's real publisher ID and token. No registry publication was attempted.
- Full diffusion sampling and VAE decoding across model families are not qualified by these tests. Conv2d padding is tested through the real patcher, not inferred from a smoke import.
- Cubemap filtering clamps within each face. Cross-face filter footprints remain a possible quality improvement.
- Long video batches retain encoded frame payloads; decoded textures are bounded. This is a preview, not a video encoder or guaranteed real-time player.
- CPU and GPU bilinear sampling differ slightly because OpenCV quantizes interpolation fractions; direction-fixture parity is tested with a 0.001 tolerance.
