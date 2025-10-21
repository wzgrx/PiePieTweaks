from .preview_image import NODE_CLASS_MAPPINGS as PreviewImageMappings
from .preview_image import NODE_DISPLAY_NAME_MAPPINGS as PreviewImageDisplayMappings
from .preview_image import WEB_DIRECTORY
from .text_concatenate import NODE_CLASS_MAPPINGS as TextConcatenateMappings
from .text_concatenate import NODE_DISPLAY_NAME_MAPPINGS as TextConcatenateDisplayMappings
from .resolution_picker import NODE_CLASS_MAPPINGS as ResolutionPickerMappings
from .resolution_picker import NODE_DISPLAY_NAME_MAPPINGS as ResolutionPickerDisplayMappings
from .resolution_from_megapixels import NODE_CLASS_MAPPINGS as ResolutionFromMPMappings
from .resolution_from_megapixels import NODE_DISPLAY_NAME_MAPPINGS as ResolutionFromMPDisplayMappings
from .lucidflux import NODE_CLASS_MAPPINGS as LucidfluxMappings
from .lucidflux import NODE_DISPLAY_NAME_MAPPINGS as LucidfluxDisplayMappings

from . import api


NODE_CLASS_MAPPINGS = {
    **PreviewImageMappings,
    **TextConcatenateMappings,
    **ResolutionPickerMappings,
    **ResolutionFromMPMappings,
    **LucidfluxMappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PreviewImageDisplayMappings,
    **TextConcatenateDisplayMappings,
    **ResolutionPickerDisplayMappings,
    **ResolutionFromMPDisplayMappings,
    **LucidfluxDisplayMappings,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
