class PiePieResolutionPicker:
    """
    PiePie - Resolution Picker
    
    Quick way to select optimal resolutions for different model types.
    Choose your model type, orientation, and get the right dimensions.
    Or go custom and enter whatever you want.
    """
    
    RESOLUTIONS = {
        "Flux": {
            "Portrait": [
                ("768x1344", 768, 1344),
                ("832x1216", 832, 1216),
                ("896x1152", 896, 1152),
                ("1024x1536", 1024, 1536),
                ("1088x1920", 1088, 1920),
            ],
            "Landscape": [
                ("1344x768", 1344, 768),
                ("1216x832", 1216, 832),
                ("1152x896", 1152, 896),
                ("1536x1024", 1536, 1024),
                ("1920x1088", 1920, 1088),
            ],
            "Square": [
                ("1024x1024", 1024, 1024),
                ("1152x1152", 1152, 1152),
            ],
        },
        "Wan": {
            "Portrait": [
                ("720x1280", 720, 1280),
                ("768x1280", 768, 1280),
                ("832x1216", 832, 1216),
                ("896x1152", 896, 1152),
                ("1024x1792", 1024, 1792),
            ],
            "Landscape": [
                ("1280x720", 1280, 720),
                ("1280x768", 1280, 768),
                ("1216x832", 1216, 832),
                ("1152x896", 1152, 896),
                ("1792x1024", 1792, 1024),
            ],
            "Square": [
                ("1024x1024", 1024, 1024),
                ("1280x1280", 1280, 1280),
            ],
        },
        "Qwen": {
            "Portrait": [
                ("768x1024", 768, 1024),
                ("832x1152", 832, 1152),
                ("896x1152", 896, 1152),
                ("1024x1536", 1024, 1536),
                ("1152x1728", 1152, 1728),
            ],
            "Landscape": [
                ("1024x768", 1024, 768),
                ("1152x832", 1152, 832),
                ("1152x896", 1152, 896),
                ("1536x1024", 1536, 1024),
                ("1728x1152", 1728, 1152),
            ],
            "Square": [
                ("1024x1024", 1024, 1024),
                ("1152x1152", 1152, 1152),
            ],
        },
        "SD1.5": {
            "Portrait": [
                ("512x768", 512, 768),
                ("448x704", 448, 704),
                ("384x640", 384, 640),
                ("512x832", 512, 832),
                ("576x896", 576, 896),
            ],
            "Landscape": [
                ("768x512", 768, 512),
                ("704x448", 704, 448),
                ("640x384", 640, 384),
                ("832x512", 832, 512),
                ("896x576", 896, 576),
            ],
            "Square": [
                ("512x512", 512, 512),
                ("576x576", 576, 576),
            ],
        },
        "SDXL": {
            "Portrait": [
                ("896x1152", 896, 1152),
                ("832x1216", 832, 1216),
                ("768x1344", 768, 1344),
                ("1024x1536", 1024, 1536),
                ("960x1728", 960, 1728),
            ],
            "Landscape": [
                ("1152x896", 1152, 896),
                ("1216x832", 1216, 832),
                ("1344x768", 1344, 768),
                ("1536x1024", 1536, 1024),
                ("1728x960", 1728, 960),
            ],
            "Square": [
                ("1024x1024", 1024, 1024),
                ("1152x1152", 1152, 1152),
            ],
        },
        "Pony": {
            "Portrait": [
                ("896x1152", 896, 1152),
                ("832x1216", 832, 1216),
                ("768x1344", 768, 1344),
                ("1024x1536", 1024, 1536),
                ("960x1728", 960, 1728),
            ],
            "Landscape": [
                ("1152x896", 1152, 896),
                ("1216x832", 1216, 832),
                ("1344x768", 1344, 768),
                ("1536x1024", 1536, 1024),
                ("1728x960", 1728, 960),
            ],
            "Square": [
                ("1024x1024", 1024, 1024),
                ("1152x1152", 1152, 1152),
            ],
        },
    }
    
    @classmethod
    def get_resolutions_for_type_and_orientation(cls, model_type, orientation):
        """Helper to get resolution list for dropdowns"""
        if model_type == "CUSTOM":
            return ["Use Custom Width/Height"]
        elif model_type == "ALL":
            if orientation == "ALL":
                all_res = set()
                for mtype in cls.RESOLUTIONS:
                    for orient in cls.RESOLUTIONS[mtype]:
                        for res_tuple in cls.RESOLUTIONS[mtype][orient]:
                            all_res.add(res_tuple[0])
                return sorted(list(all_res), key=lambda x: int(x.split('x')[0].split()[0]))
            else:
                all_res = set()
                for mtype in cls.RESOLUTIONS:
                    if orientation in cls.RESOLUTIONS[mtype]:
                        for res_tuple in cls.RESOLUTIONS[mtype][orientation]:
                            all_res.add(res_tuple[0])
                return sorted(list(all_res), key=lambda x: int(x.split('x')[0].split()[0]))
        else:
            if orientation == "ALL":
                all_res = []
                for orient in cls.RESOLUTIONS[model_type]:
                    for res_tuple in cls.RESOLUTIONS[model_type][orient]:
                        all_res.append(res_tuple[0])
                return all_res
            else:
                return [res[0] for res in cls.RESOLUTIONS[model_type][orientation]]
    
    @classmethod
    def INPUT_TYPES(s):
        # Start with default resolutions (ALL + ALL)
        default_resolutions = s.get_resolutions_for_type_and_orientation("ALL", "ALL")
        
        return {
            "required": {
                "type": (["ALL", "Flux", "Wan", "Qwen", "SD1.5", "SDXL", "Pony", "CUSTOM"], 
                        {"default": "ALL"}),
                "orientation": (["ALL", "Portrait", "Landscape", "Square"], 
                               {"default": "ALL"}),
                "resolution": (default_resolutions, {"default": default_resolutions[0]}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "PiePieDesign"
    
    def get_resolution(self, type, orientation, resolution, custom_width=1024, custom_height=1024):
        """
        Returns width and height based on selected options
        
        If CUSTOM is selected, uses custom_width and custom_height.
        Otherwise parses the selected resolution string.
        """
        
        # Handle custom resolution
        if type == "CUSTOM":
            return (custom_width, custom_height)
        
        # Extract just the numbers
        res_str = resolution.split()[0]  # Get "1024x1024" part
        width, height = map(int, res_str.split('x'))
        
        return (width, height)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "PiePieResolutionPicker": PiePieResolutionPicker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiePieResolutionPicker": "PiePie - Resolution Picker"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]