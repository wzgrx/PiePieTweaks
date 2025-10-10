# PiePieTweaks

Custom nodes for ComfyUI by PiePieDesign

## Nodes

### PiePie - Preview Image

A preview node that gives you control over when images get saved.

**The Problem It Solves:**
- Combines behavior of Preview Image and Save Image core nodes
- Allows for ad-hoc saving to follow the Save Image prefix formats
- Key use case: you have a lot of runs with mixed quality results. You only want to save the good ones and not the bad ones. -> Use "Manual Save" mode
- Key use case: you want to save all the runs just like the Save Image node. -> Use "Always Save" mode

**How It Works:**

Two modes to choose from:

1. **Always save mode** - Works exactly like Save Image
   - Every generation automatically saves to your output folder

2. **Manual save mode** - Only saves when you click the button
   - Generates and shows preview
   - Nothing gets saved until you click "💾 Manual Save"
   - Bad generations? Just re-generate
   - Good generation? Click save and it goes to your output folder with the same prefix you would have had with the Save Image node

---

### PiePie - Resolution Picker

Quick way to pick optimal resolutions for different model types.

**The Problem It Solves:**
- No more guessing which resolution works best for your model
- No more manually typing in width/height values
- Extensable with the js/resolutions.js file

**How It Works:**

1. Choose your model type (Flux, SDXL, SD1.5, Pony, etc.)
2. Choose orientation (Portrait, Landscape, Square)
3. Pick from the filtered list of optimal resolutions
4. Or select CUSTOM and enter whatever dimensions you want

The dropdown dynamically updates based on your selections, showing only relevant resolutions.

---

### PiePie - Resolution from Megapixels

Find the closest suggested resolution to your target megapixel count.

**The Problem It Solves:**
- You know you want "around 1.5 megapixels" but don't want to calculate dimensions
- Key use case: you're iterating fast and want consistent megapixel counts without manually picking resolutions
- Optional use case: you want to stay under a certain megapixel limit for VRAM reasons

**How It Works:**

1. Set your target megapixels (e.g., 1.5)
2. Choose model type and orientation filters (or leave as ALL)
3. Toggle "do not exceed" if you want to enforce a hard limit
4. Node outputs the closest matching width, height, and actual megapixels

---

### PiePie - Text Concatenate

Simple text concatenation with a separator.

**The Problem It Solves:**
- Joining multiple text inputs without messy workarounds
- Key use case: combining prompts, filenames, or tags from different sources

**How It Works:**

Takes multiple text inputs and joins them with your chosen separator (space, comma, newline, etc.).
Allows for manual entry or inputs.


## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI
2. Click the "Manager" button
3. Click "Install Custom Nodes"
4. Search for "PiePieTweaks"
5. Click Install
6. Restart ComfyUI

### Method 2: Manual Install

Clone this repository into ComfyUI/custom_nodes

## Contributing

Found a bug? Have a feature idea? Open an issue!

Pull requests welcome. Please keep it simple and well-commented.

## License

MIT License

## Credits

Made by PiePieDesign