class PiePieTextConcatenate:  
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delimiter": ([
                    "Space", 
                    "Comma + Space", 
                    "Comma",
                    "Custom"
                ], {"default": "Comma + Space"}),
                "newline_after_each": (["No", "Yes"], {"default": "No"}),
                "custom_delimiter": ("STRING", {"default": " | "}),
            },
            "optional": {
                "text1": ("STRING", {"default": "", "multiline": True}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "text3": ("STRING", {"default": "", "multiline": True}),
                "text4": ("STRING", {"default": "", "multiline": True}),
                "text5": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concatenate_text"
    OUTPUT_NODE = True  # This allows it to execute without being connected
    CATEGORY = "PiePieDesign"

    def concatenate_text(self, delimiter, newline_after_each, custom_delimiter,
                        text1="", text2="", text3="", text4="", text5=""):
                        
        texts = []
        for text in [text1, text2, text3, text4, text5]:
            if text and text.strip():
                texts.append(text)
        
        # If nothing was provided, return empty string
        if not texts:
            result = ""
        else:

            if delimiter == "Space":
                delim = " "
            elif delimiter == "Comma + Space":
                delim = ", "
            elif delimiter == "Comma":
                delim = ","
            else:  # Custom
                delim = custom_delimiter if custom_delimiter is not None else " | "
            
            if newline_after_each == "Yes":
                delim = delim + "\n"
            
            # Join all the texts together
            result = delim.join(texts)
        
        # Return with UI update
        return {"ui": {"string": [result]}, "result": (result,)}


# Registration
NODE_CLASS_MAPPINGS = {
    "PiePieTextConcatenate": PiePieTextConcatenate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiePieTextConcatenate": "PiePie - Text Concatenate"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
