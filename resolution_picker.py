from .resolutions_db import RESOLUTIONS

class PiePieResolutionPicker:
   
    @classmethod
    def get_resolutions_for_type_and_orientation(cls, model_type, orientation):
        if model_type == "CUSTOM":
            return ["Use Custom Width/Height"]
        elif model_type == "ALL":
            if orientation == "ALL":
                all_res = set()
                for mtype in RESOLUTIONS:
                    for orient in RESOLUTIONS[mtype]:
                        for width, height in RESOLUTIONS[mtype][orient]:
                            all_res.add(f"{width}x{height}")
                return sorted(list(all_res), key=lambda x: int(x.split('x')[0]))
            else:
                all_res = set()
                for mtype in RESOLUTIONS:
                    if orientation in RESOLUTIONS[mtype]:
                        for width, height in RESOLUTIONS[mtype][orientation]:
                            all_res.add(f"{width}x{height}")
                return sorted(list(all_res), key=lambda x: int(x.split('x')[0]))
        else:
            if orientation == "ALL":
                all_res = []
                for orient in RESOLUTIONS[model_type]:
                    for width, height in RESOLUTIONS[model_type][orient]:
                        all_res.append(f"{width}x{height}")
                return all_res
            else:
                return [f"{w}x{h}" for w, h in RESOLUTIONS[model_type][orientation]]
    
    @classmethod
    def INPUT_TYPES(s):
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
        if type == "CUSTOM":
            return (custom_width, custom_height)
        
        width, height = map(int, resolution.split('x'))
        
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