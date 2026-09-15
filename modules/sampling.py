"""Pixel-center sampling: periodic longitude, clamped latitude, bounded source tiles."""
import cv2
import numpy as np
import torch


FILTERS = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
           "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


def remap(image, x, y, method="lanczos", wrap_x=True):
    if method not in FILTERS:
        raise ValueError(f"Unknown interpolation: {method}")
    h, w = image.shape[:2]
    shape = x.shape
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    x = np.mod(x, w) if wrap_x else np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    # Each remap source and destination stays below OpenCV's signed-short limit.
    tile = 256
    halo = 4
    keys = (y.astype(np.int64) // tile) * ((w + tile - 1) // tile) + x.astype(np.int64) // tile
    order = np.argsort(keys, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
    channels = image.shape[2:] or ()
    out = np.empty((len(x),) + channels, dtype=image.dtype)
    for start, end in zip(cuts[:-1], cuts[1:]):
        ids = order[start:end]
        if not len(ids):
            continue
        x0 = int(x[ids[0]]) // tile * tile
        y0 = int(y[ids[0]]) // tile * tile
        xx = np.arange(x0 - halo, min(x0 + tile, w) + halo)
        yy = np.clip(np.arange(y0 - halo, min(y0 + tile, h) + halo), 0, h - 1)
        xx = np.mod(xx, w) if wrap_x else np.clip(xx, 0, w - 1)
        source = image[yy[:, None], xx[None, :]]
        for offset in range(0, len(ids), 32000):
            idx = ids[offset:offset + 32000]
            values = cv2.remap(source, (x[idx] - x0 + halo)[:, None],
                               (y[idx] - y0 + halo)[:, None], FILTERS[method],
                               borderMode=cv2.BORDER_REPLICATE)
            out[idx] = values.reshape((len(idx),) + channels)
    return out.reshape(shape + channels)


def torch_remap(image, x, y, method="bilinear", padded=None):
    if method not in ("bilinear", "nearest"):
        raise ValueError("Torch sampling supports bilinear and nearest; use CPU for bicubic/Lanczos")
    h, w, _ = image.shape
    # Two halo columns make interpolation across +/-180 degrees periodic.
    source = padded if padded is not None else torch.cat((image[:, -1:], image, image[:, :1]), dim=1)
    x = torch.remainder(x, w) + 1
    y = y.clamp(0, h - 1)
    grid = torch.stack((2 * x / (w + 1) - 1,
                        2 * y / max(1, h - 1) - 1), dim=-1)[None]
    result = torch.nn.functional.grid_sample(source.permute(2, 0, 1)[None], grid,
                                             mode=method, padding_mode="border", align_corners=True)
    return result[0].permute(1, 2, 0)
