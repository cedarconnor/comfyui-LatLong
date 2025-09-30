from .nodes import (
    EquirectangularRotate,
    EquirectangularCrop180,
    EquirectangularCropSquare,
    EquirectangularProcessor_Combined,
    EquirectangularPerspectiveExtract,
    EquirectangularToCubemap,
    CubemapToEquirectangular,
    EquirectangularMirrorFlip,
    EquirectangularResize,
    EquirectangularRotatePreset,
    CubemapFacesExtract,
)


NODE_CLASS_MAPPINGS = {
    "Equirectangular Rotate": EquirectangularRotate,
    "Equirectangular Crop 180": EquirectangularCrop180,
    "Equirectangular Crop Square": EquirectangularCropSquare,
    "Equirectangular Processor": EquirectangularProcessor_Combined,
    "Equirectangular Perspective Extract": EquirectangularPerspectiveExtract,
    "Equirectangular To Cubemap": EquirectangularToCubemap,
    "Cubemap To Equirectangular": CubemapToEquirectangular,
    "Equirectangular Mirror Flip": EquirectangularMirrorFlip,
    "Equirectangular Resize": EquirectangularResize,
    "Equirectangular Rotate Preset": EquirectangularRotatePreset,
    "Cubemap Faces Extract": CubemapFacesExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Equirectangular Rotate": "Equirectangular Rotate",
    "Equirectangular Crop 180": "Equirectangular Crop 180",
    "Equirectangular Crop Square": "Equirectangular Crop Square",
    "Equirectangular Processor": "Equirectangular Processor (All-in-One)",
    "Equirectangular Perspective Extract": "Equirectangular Perspective Extract",
    "Equirectangular To Cubemap": "Equirectangular → Cubemap (3x2)",
    "Cubemap To Equirectangular": "Cubemap (3x2) → Equirectangular",
    "Equirectangular Mirror Flip": "Equirectangular Mirror/Flip",
    "Equirectangular Resize": "Equirectangular Resize",
    "Equirectangular Rotate Preset": "Equirectangular Rotate (Preset)",
    "Cubemap Faces Extract": "Cubemap Faces Extract",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
