from .resolutions_db import RESOLUTIONS

class PiePieResolutionFromMegapixels:

    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "type": (["ALL", "Flux", "Wan", "Qwen", "SD1.5", "SDXL", "Pony"], 
                        {"default": "ALL"}),
                "orientation": (["ALL", "Portrait", "Landscape", "Square"], 
                               {"default": "ALL"}),
                "do_not_exceed": (["No", "Yes"], {"default": "No"}),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "megapixels")
    FUNCTION = "find_resolution"
    CATEGORY = "PiePieDesign"
    
    def find_resolution(self, target_megapixels, type, orientation, do_not_exceed):

        
        exceed_limit = (do_not_exceed == "Yes")
        
        resolutions = self._get_filtered_resolutions(type, orientation)
        
        if not resolutions:
            print(f"[PiePie Resolution from MP] ERROR: No resolutions found for type={type}, orientation={orientation}")
            return (1024, 1024, 1.048576)
        
        res_with_mp = [(w, h, self._calculate_megapixels(w, h)) for w, h in resolutions]
        
        if exceed_limit:
            valid_resolutions = [(w, h, mp) for w, h, mp in res_with_mp if mp <= target_megapixels]
            
            if not valid_resolutions:
                width, height, actual_mp = min(res_with_mp, key=lambda x: x[2])
                print(f"[PiePie Resolution from MP] No resolutions ≤ {target_megapixels:.2f}MP found with current filters (type={type}, orientation={orientation}). Using smallest available: {width}x{height} ({actual_mp:.2f}MP)")
                return (width, height, actual_mp)
            
            width, height, actual_mp = min(valid_resolutions, key=lambda x: abs(x[2] - target_megapixels))
        else:
            width, height, actual_mp = min(res_with_mp, key=lambda x: abs(x[2] - target_megapixels))
        
        return (width, height, actual_mp)
    
    def _get_filtered_resolutions(self, model_type, orientation):

        results = []
        
        model_types = RESOLUTIONS.keys() if model_type == "ALL" else [model_type]
        
        for mtype in model_types:
            if mtype not in RESOLUTIONS:
                continue
            
            orientations = RESOLUTIONS[mtype].keys() if orientation == "ALL" else [orientation]
            
            for orient in orientations:
                if orient in RESOLUTIONS[mtype]:
                    results.extend(RESOLUTIONS[mtype][orient])
        
        seen = set()
        unique_results = []
        for res in results:
            if res not in seen:
                seen.add(res)
                unique_results.append(res)
        
        return unique_results
    
    def _calculate_megapixels(self, width, height):
        return (width * height) / 1_000_000


NODE_CLASS_MAPPINGS = {
    "PiePieResolutionFromMegapixels": PiePieResolutionFromMegapixels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiePieResolutionFromMegapixels": "PiePie - Resolution from Megapixels"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]