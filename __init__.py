from .nodes import EquirectangularRotate, EquirectangularCrop180, EquirectangularProcessor_Combined


NODE_CLASS_MAPPINGS = {
    "Equirectangular Rotate": EquirectangularRotate,
    "Equirectangular Crop 180": EquirectangularCrop180, 
    "Equirectangular Processor": EquirectangularProcessor_Combined,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Equirectangular Rotate": "Equirectangular Rotate",
    "Equirectangular Crop 180": "Equirectangular Crop 180",
    "Equirectangular Processor": "Equirectangular Processor (All-in-One)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']