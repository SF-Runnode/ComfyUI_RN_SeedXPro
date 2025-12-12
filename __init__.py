from .SeedXPro import RN_SeedXPro_Translator

NODE_CLASS_MAPPINGS = {
    "RN SeedXPro Translator": RN_SeedXPro_Translator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RN SeedXPro Translator": "RunNode SeedXPro Translater",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']