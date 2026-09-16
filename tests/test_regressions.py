import importlib
import os
import unittest
import numpy as np
import torch
from run_node_smoke_tests import stub_comfy_modules, load_latlong_nodes_package

stub_comfy_modules()
n = load_latlong_nodes_package(os.path.dirname(os.path.dirname(__file__)))
w = importlib.import_module('latlong.workflow_nodes')
P = n.EquirectangularProcessor


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.a = np.random.default_rng(4).random((32, 64, 3), dtype=np.float32)

    def test_identity(self):
        np.testing.assert_array_equal(P.rotate_equirectangular(self.a), self.a)
        np.testing.assert_array_equal(P.torch_rotate_equirectangular(torch.from_numpy(self.a)), self.a)

    def test_matrix_convention(self):
        expected = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
        np.testing.assert_allclose(P.create_rotation_matrix(90, 0, 90), expected, atol=1e-7)
        np.testing.assert_allclose(P._torch_rotation_matrix(45, 30, 15, 'cpu', torch.float32), P.create_rotation_matrix(45, 30, 15), atol=1e-7)

    def test_backend_parity(self):
        # Smooth direction-encoded fixture isolates geometry from filter quantization.
        yy, xx = np.mgrid[:64, :128]
        lat, lon = P.equirectangular_to_spherical(xx, yy, 128, 64)
        a = (np.stack(P.spherical_to_cartesian(lat, lon), -1).astype(np.float32) + 1) / 2
        cpu = P.rotate_equirectangular(a, 45, 30, 15, interpolation='bilinear')
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            result = P.torch_rotate_equirectangular(torch.from_numpy(a).to(device), 45, 30, 15).cpu().numpy()
            np.testing.assert_allclose(result, cpu, atol=.001)

    def test_seam_and_poles(self):
        from latlong.modules.sampling import torch_remap
        a = np.zeros((8, 16, 3), np.float32); a[:, 0] = 1
        x, y = np.array([[15.5]], np.float32), np.array([[4]], np.float32)
        np.testing.assert_allclose(P.interpolate_image(a, x, y, 'bilinear'), .5)
        np.testing.assert_allclose(torch_remap(torch.from_numpy(a), torch.from_numpy(x), torch.from_numpy(y)), .5)
        a[:] = 0; a[-1] = 1
        for mode in ['bicubic', 'lanczos']:
            np.testing.assert_allclose(P.interpolate_image(a, np.array([[4]]), np.array([[.25]]), mode), 0)

    def test_large_source(self):
        a = np.ones((2, 32768, 3), np.float32)
        np.testing.assert_allclose(P.interpolate_image(a, np.array([[32767.5]]), np.array([[.5]])), 1, atol=1e-6)

    def test_perspective_cardinals(self):
        for yaw, expected in [(0, 512), (90, 768), (-90, 256)]:
            x, y = P._cached_perspective_maps(1024, 512, 65, 65, yaw, 0, 0, 90)
            self.assertAlmostEqual(float(x[32,32]), expected, places=4)
            self.assertAlmostEqual(float(y[32,32]), 256, places=4)

    def test_projection_rejects_undefined_geometry(self):
        for fov in [0, 180, float('nan')]:
            with self.assertRaises(ValueError): P._cached_perspective_maps(64,32,16,16,0,0,0,fov)

    def test_tiled(self):
        direct = P.rotate_equirectangular(self.a, 21, 12, 7, use_tiling=False)
        tiled = P.rotate_equirectangular(self.a, 21, 12, 7, use_tiling=True, tile_size=9)
        np.testing.assert_allclose(direct, tiled, atol=1e-6)

    def test_filters(self):
        atlas = np.random.default_rng(3).random((16, 24, 3), dtype=np.float32)
        a = P.cubemap_to_equirectangular(atlas, 64, 32, interpolation='bilinear')
        for mode in ['bicubic', 'lanczos']:
            self.assertGreater(np.max(abs(a - P.cubemap_to_equirectangular(atlas, 64, 32, interpolation=mode))), .01)

    def test_blend_boundaries(self):
        a = np.broadcast_to(np.linspace(0, 1, 100, dtype=np.float32)[None,:,None], (10,100,3)).copy()
        for mode in ['cosine','linear','smooth']:
            b = P.blend_edges(a, 10, mode)
            np.testing.assert_allclose(b[:,0], b[:,-1])
            np.testing.assert_allclose(b[:,9:11], a[:,9:11])
            np.testing.assert_allclose(b[:,89:91], a[:,89:91])


class NodeTests(unittest.TestCase):
    def setup_patch(self, **extra):
        args = dict(flat_image=torch.ones(1,32,64,3), canvas_width=128, canvas_height=64,
                    force_2by1_aspect=True, placement_mode='perspective', scale=1., translate_x=0,
                    translate_y=0, yaw=0., pitch=0., fov=90., feather_size=0)
        args.update(extra)
        return n.LatLongOutpaintSetup().setup(**args)

    def test_scale(self):
        _, a, _ = self.setup_patch()
        _, b, _ = self.setup_patch(scale=2.)
        self.assertGreater(float((1-b).sum()), float((1-a).sum()) * 1.5)

    def test_blend_modes(self):
        _, _, context = self.setup_patch(placement_mode='2d_composite', feather_size=8)
        generated = torch.full((1,64,128,3), .25)
        outputs = [n.LatLongOutpaintStitch().stitch(generated, context, mode)[0] for mode in ['alpha','hard','overlay']]
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        self.assertFalse(torch.equal(outputs[0], outputs[2]))

    def test_wrapped_placement_and_stitch(self):
        a, m, c = self.setup_patch(placement_mode='2d_composite', translate_x=64)
        self.assertGreater(float(a[:,:,0].sum()), 0)
        self.assertGreater(float(a[:,:,-1].sum()), 0)
        result = n.LatLongOutpaintStitch().stitch(torch.zeros_like(a), c, 'alpha')[0]
        np.testing.assert_allclose(a, result)

    def test_grayscale_preview(self):
        a = torch.zeros(1,16,32,1)
        self.assertIn('pano_image', n.PanoramaViewerNode().view_pano(a)['ui'])
        self.assertIn('pano_video_frames', n.PanoramaVideoViewerNode().view_video_pano(a)['ui'])

    def test_nonfinite(self):
        with self.assertRaises(ValueError): n.EquirectangularRotate().rotate_equirectangular(torch.full((1,2,4,3), float('nan')))

    def test_filter_dispatch(self):
        self.assertFalse(n._use_gpu('auto', 'lanczos'))
        with self.assertRaises(ValueError): n._use_gpu('gpu', 'lanczos')

    def test_padding_switch(self):
        conv = torch.nn.Conv2d(1,1,3,padding=1,bias=False); conv.weight.data.fill_(1)
        x = torch.arange(25,dtype=torch.float32).reshape(1,1,5,5)
        n._apply_circular_conv2d_padding(conv,x_axis_only=True)
        n._apply_circular_conv2d_padding(conv,x_axis_only=False)
        expected = torch.nn.functional.conv2d(torch.nn.functional.pad(x,(1,1,1,1),mode='circular'),conv.weight)
        torch.testing.assert_close(conv(x),expected)

    def test_hdr(self):
        a = torch.full((1,16,32,3), 4.)
        a[:,:,:16] = -1
        out = w.LatLongFloatProcessor().process(a,'rotate',0,0,0,90,32,16,'bilinear')[0]
        torch.testing.assert_close(out,a)
        preview = w.LatLongToneMap().process(a,0)[0]
        self.assertLessEqual(float(preview.max()),1)
        self.assertEqual(float(a.max()),4)

    def test_projection_roundtrip(self):
        yy, xx = np.mgrid[:128,:256]
        a = torch.from_numpy(np.stack([xx/256, yy/128, np.ones_like(xx)*.5], -1).astype(np.float32))[None]
        patch, mask, context = w.LatLongExtractProjection().extract(a,0,0,0,70,65,65)
        out = w.LatLongReinsertProjection().stitch(a,patch,context,4)[0]
        self.assertLess(float((out-a).abs().max()),.015)
        self.assertEqual(mask.shape,(1,128,256))

    def test_animation(self):
        a = torch.rand(1,16,32,3)
        out = w.LatLongAnimatedRotation().rotate(a,'[{"frame":0,"yaw":0},{"frame":2,"yaw":90}]',3,'nearest')[0]
        torch.testing.assert_close(out[0],a[0])
        np.testing.assert_allclose(out[-1],P.rotate_equirectangular(a[0].numpy(),90,interpolation='nearest'))

    def test_diagnostics(self):
        _, report = w.LatLongDiagnostics().analyze(torch.ones(1,8,16,3),3)
        self.assertIn('"wrap_mean": 0.0',report)

    def test_cube_presets(self):
        a = torch.from_numpy(np.random.default_rng(17).random((1,32,64,3), dtype=np.float32))
        expected = n.CubemapToEquirectangularFlexible().to_equirectangular_flexible(
            n.EquirectangularToCubemapFlexible().to_cubemap_flexible(a,16,interpolation='bilinear')[0],
            output_width=64,output_height=32,interpolation='bilinear')[0]
        for preset in w.LatLongCubemapPreset.PRESETS:
            for flip in [False, True]:
                node = w.LatLongCubemapPreset()
                cube = node.convert(a,'export',preset,16,64,flip)[0]
                out = node.convert(cube,'import',preset,16,64,flip)[0]
                torch.testing.assert_close(expected,out,atol=1e-5,rtol=1e-5)

    def test_mono_transforms(self):
        a = torch.ones(1,32,64,1)
        self.assertEqual(n.EquirectangularResize().resize(a,128)[0].shape,(1,64,128,1))
        self.assertEqual(n.EquirectangularCrop180().crop_to_180(a,64)[0].shape,(1,64,64,1))

    def test_comparison_sequence(self):
        before = torch.zeros(1,16,32,3)
        after = torch.ones(3,16,32,3)
        ui = w.LatLongComparePanorama().compare(before,after,64,12)['ui']
        self.assertEqual(len(ui['pano_video_frames']),3)
        self.assertEqual(len(ui['compare_frames']),3)
        self.assertEqual(ui['fps'],['12'])


if __name__ == '__main__': unittest.main()
