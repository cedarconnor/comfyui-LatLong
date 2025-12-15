import os
import sys
import types
import importlib.util
import numpy as np
import torch


def stub_comfy_modules():
    comfy = types.ModuleType('comfy')
    utils = types.ModuleType('comfy.utils')

    class ProgressBar:
        def __init__(self, total):
            self.total = total
        def update(self, value):
            pass

    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    sys.modules['comfy'] = comfy
    sys.modules['comfy.utils'] = utils

    folder_paths = types.ModuleType('folder_paths')
    sys.modules['folder_paths'] = folder_paths


def load_latlong_nodes_package(repo_root: str):
    pkg = types.ModuleType('latlong')
    pkg.__path__ = [repo_root]
    sys.modules['latlong'] = pkg

    # Create subpackage latlong.modules
    modules_pkg = types.ModuleType('latlong.modules')
    modules_pkg.__path__ = [os.path.join(repo_root, 'modules')]
    sys.modules['latlong.modules'] = modules_pkg

    # Load latlong.modules.equirectangular_processor
    proc_path = os.path.join(repo_root, 'modules', 'equirectangular_processor.py')
    spec_proc = importlib.util.spec_from_file_location('latlong.modules.equirectangular_processor', proc_path)
    mod_proc = importlib.util.module_from_spec(spec_proc)
    sys.modules['latlong.modules.equirectangular_processor'] = mod_proc
    spec_proc.loader.exec_module(mod_proc)

    # Load latlong.nodes (uses relative import to modules)
    nodes_path = os.path.join(repo_root, 'nodes.py')
    spec_nodes = importlib.util.spec_from_file_location('latlong.nodes', nodes_path)
    mod_nodes = importlib.util.module_from_spec(spec_nodes)
    mod_nodes.__package__ = 'latlong'  # for relative imports
    sys.modules['latlong.nodes'] = mod_nodes
    spec_nodes.loader.exec_module(mod_nodes)

    return mod_nodes


def make_test_image(h=512, w=1024, c=3):
    # Synthetic equirectangular-like gradient
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    img = np.stack([
        xv,
        yv,
        0.5 * np.ones_like(xv, dtype=np.float32)
    ], axis=-1)
    if c == 4:
        alpha = np.ones((h, w, 1), dtype=np.float32)
        img = np.concatenate([img, alpha], axis=-1)
    return img


def to_batch_tensor(np_img, batch=2):
    t = torch.from_numpy(np_img)
    t = t.unsqueeze(0).repeat(batch, 1, 1, 1)
    return t


def assert_range01(arr, name, atol=1e-3):
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mn < -atol or mx > 1.0 + atol:
        raise AssertionError(f"{name} out of [0,1] range: min={mn} max={mx}")


def run_tests():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stub_comfy_modules()
    nodes = load_latlong_nodes_package(repo_root)

    base = make_test_image(512, 1024, 3)
    batch = to_batch_tensor(base, batch=2)

    # Rotate CPU
    rot_node = nodes.EquirectangularRotate()
    out_rot_cpu, = rot_node.rotate_equirectangular(batch, yaw_rotation=45.0, pitch_rotation=-10.0, roll_rotation=15.0, horizon_offset=5.0, interpolation='lanczos', backend='cpu')
    assert out_rot_cpu.shape == batch.shape
    assert_range01(out_rot_cpu.numpy(), 'rotate_cpu')

    # Rotate GPU (optional)
    if torch.cuda.is_available():
        out_rot_gpu, = rot_node.rotate_equirectangular(batch, yaw_rotation=15.0, pitch_rotation=5.0, roll_rotation=0.0, horizon_offset=0.0, interpolation='bilinear', backend='gpu')
        if out_rot_gpu.shape != batch.shape:
            raise AssertionError(f"GPU rotate shape mismatch: got {tuple(out_rot_gpu.shape)}, expected {tuple(batch.shape)}")
        assert_range01(out_rot_gpu.numpy(), 'rotate_gpu')

    # Crop 180 default
    crop180 = nodes.EquirectangularCrop180()
    out_crop_def, = crop180.crop_to_180(batch, output_width=800, output_height=400, maintain_aspect=False, interpolation='bicubic')
    assert out_crop_def.shape[1] == 400 and out_crop_def.shape[2] == 800
    assert_range01(out_crop_def.numpy(), 'crop180_default')

    # Crop 120° centered at 90° with maintain_aspect
    out_crop_custom, = crop180.crop_to_180(batch, output_width=600, output_height=300, maintain_aspect=True, center_longitude_deg=90.0, fov_degrees=120.0, interpolation='lanczos')
    assert out_crop_custom.shape[2] == 600
    assert_range01(out_crop_custom.numpy(), 'crop180_custom')

    # Perspective CPU
    persp = nodes.EquirectangularPerspectiveExtract()
    out_persp_cpu, = persp.extract_perspective(batch, yaw_rotation=0.0, pitch_rotation=0.0, roll_rotation=0.0, fov_degrees=90.0, output_width=512, output_height=512, interpolation='lanczos', backend='cpu')
    assert out_persp_cpu.shape[1] == 512 and out_persp_cpu.shape[2] == 512
    assert_range01(out_persp_cpu.numpy(), 'perspective_cpu')

    # Perspective GPU (optional)
    if torch.cuda.is_available():
        out_persp_gpu, = persp.extract_perspective(batch, yaw_rotation=30.0, pitch_rotation=-10.0, roll_rotation=5.0, fov_degrees=75.0, output_width=640, output_height=480, interpolation='bilinear', backend='gpu')
        assert out_persp_gpu.shape[1] == 480 and out_persp_gpu.shape[2] == 640
        assert_range01(out_persp_gpu.numpy(), 'perspective_gpu')

    # Cubemap
    to_cube = nodes.EquirectangularToCubemap()
    out_cube, = to_cube.to_cubemap(batch, face_size=64, interpolation='lanczos')
    assert out_cube.shape[1] == 64*2 and out_cube.shape[2] == 64*3
    assert_range01(out_cube.numpy(), 'cubemap')

    # Cubemap (flexible) -> stack -> back to equirectangular
    to_cube_flex = nodes.EquirectangularToCubemapFlexible()
    cube_stack, = to_cube_flex.to_cubemap_flexible(
        batch, face_size=64, cube_format="stack", face_order="F,R,B,L,U,D", interpolation="bilinear"
    )
    assert cube_stack.shape[0] == batch.shape[0] * 6
    assert cube_stack.shape[1] == 64 and cube_stack.shape[2] == 64
    assert_range01(cube_stack.numpy(), "cubemap_stack")

    to_equi_flex = nodes.CubemapToEquirectangularFlexible()
    equi_from_stack, = to_equi_flex.to_equirectangular_flexible(
        cube_stack,
        cube_format="stack",
        face_order="F,R,B,L,U,D",
        output_width=batch.shape[2],
        output_height=batch.shape[1],
        interpolation="bilinear",
    )
    assert equi_from_stack.shape == batch.shape
    assert_range01(equi_from_stack.numpy(), "equi_from_stack")

    # Dice layout round-trip
    cube_dice, = to_cube_flex.to_cubemap_flexible(
        batch, face_size=64, cube_format="dice", interpolation="nearest"
    )
    assert cube_dice.shape[1] == 64 * 3 and cube_dice.shape[2] == 64 * 4
    equi_from_dice, = to_equi_flex.to_equirectangular_flexible(
        cube_dice,
        cube_format="dice",
        output_width=batch.shape[2],
        output_height=batch.shape[1],
        interpolation="bilinear",
    )
    assert equi_from_dice.shape == batch.shape
    assert_range01(equi_from_dice.numpy(), "equi_from_dice")

    # Horizon layout round-trip
    cube_horizon, = to_cube_flex.to_cubemap_flexible(
        batch,
        face_size=64,
        cube_format="horizon",
        face_order="F,R,B,L,U,D",
        interpolation="bilinear",
    )
    assert cube_horizon.shape[1] == 64 and cube_horizon.shape[2] == 64 * 6
    equi_from_horizon, = to_equi_flex.to_equirectangular_flexible(
        cube_horizon,
        cube_format="horizon",
        face_order="F,R,B,L,U,D",
        output_width=batch.shape[2],
        output_height=batch.shape[1],
        interpolation="bilinear",
    )
    assert equi_from_horizon.shape == batch.shape
    assert_range01(equi_from_horizon.numpy(), "equi_from_horizon")

    # Stack/split utilities
    split_stack = nodes.SplitCubemapFacesNode()
    f, r, b, l, u, d = split_stack.split_faces(cube_stack, face_order="F,R,B,L,U,D")
    assert f.shape[0] == batch.shape[0] and f.shape[1] == 64 and f.shape[2] == 64

    stack_faces = nodes.StackCubemapFacesNode()
    cube_stack2, = stack_faces.stack_faces(
        f, r, b, l, u, d, face_order="F,R,B,L,U,D"
    )
    assert cube_stack2.shape == cube_stack.shape

    # Combined processor basics
    comb = nodes.EquirectangularProcessor_Combined()
    out_comb, = comb.process_equirectangular(batch, yaw_rotation=0.0, pitch_rotation=0.0, roll_rotation=0.0, horizon_offset=0.0, crop_to_180=False, crop_to_square=True, output_width=512, output_height=256, interpolation='bilinear')
    h, w = out_comb.shape[1:3]
    assert h == w, 'square crop expected'
    assert_range01(out_comb.numpy(), 'combined_square')

    # Seam mask + roll utilities
    seam = nodes.LatLongCreateSeamMask()
    seam_mask, = seam.run(batch, frac_width=0.1, pixel_width=0, feather=8, roll_x_by_50_percent=False)
    assert seam_mask.shape == (batch.shape[0], batch.shape[1], batch.shape[2])
    assert_range01(seam_mask.numpy(), "seam_mask")

    roll_img = nodes.LatLongRollImage()
    rolled_img, = roll_img.roll(batch, roll_x_by_50_percent=True)
    assert rolled_img.shape == batch.shape

    roll_mask = nodes.LatLongRollMask()
    rolled_mask, = roll_mask.roll(seam_mask, roll_x_by_50_percent=True)
    assert rolled_mask.shape == seam_mask.shape

    # Pole mask (face + equirectangular)
    pole = nodes.LatLongCreatePoleMask()
    pole_face_mask, = pole.run(f, circle_radius=0.25, pixel_radius=0, feather=4, mode="face")
    assert pole_face_mask.shape == (batch.shape[0], 64, 64)
    assert_range01(pole_face_mask.numpy(), "pole_face_mask")

    pole_equi_mask, = pole.run(batch, circle_radius=0.25, pixel_radius=0, feather=4, mode="equirectangular", face_size=64)
    assert pole_equi_mask.shape == (batch.shape[0], batch.shape[1], batch.shape[2])
    assert_range01(pole_equi_mask.numpy(), "pole_equi_mask")

    print('All node smoke tests passed.')


if __name__ == '__main__':
    run_tests()
