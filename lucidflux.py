import numpy as np
import torch
import os
from omegaconf import OmegaConf
import folder_paths
import gc
import nodes

from . import model_loader_utils
from . import inference
from .src.flux import align_color

# Extract the functions we need from src -- create an issue to improve this later
tensor2pillist_upscale = model_loader_utils.tensor2pillist_upscale
load_lucidflux_model = inference.load_lucidflux_model
lucidflux_inference = inference.lucidflux_inference
get_cond = inference.get_cond
get_cond_from_embeddings = inference.get_cond_from_embeddings
preprocess_data_cached = inference.preprocess_data_cached
print_memory_status = inference.print_memory_status
aggressive_cleanup = inference.aggressive_cleanup
wavelet_reconstruction = align_color.wavelet_reconstruction

MAX_SEED = np.iinfo(np.int32).max

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_LucidFlux_current_path = os.path.join(folder_paths.models_dir, "LucidFlux")
if not os.path.exists(weigths_LucidFlux_current_path):
    os.makedirs(weigths_LucidFlux_current_path)
folder_paths.add_model_folder_path("LucidFlux", weigths_LucidFlux_current_path)


class PiePie_Lucidflux:

    _model_cache = {}
    _swinir_cache = {}
    _redux_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_model": ("MODEL",),
                "LucidFlux": (["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "lucid" in i.lower()],),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "swinir": (["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "swinir" in i.lower()],),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": nodes.MAX_RESOLUTION,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": nodes.MAX_RESOLUTION,
                    "step": 64
                }),
                "CLIP_VISION": ("CLIP_VISION",),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_SEED
                }),
                "cfg": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "use_embeddings": ("BOOLEAN", {"default": True}),
                "prompt_embeddings": (["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "prompt" in i.lower() or i.endswith(".pt")],),
                "enable_offload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "PiePie"
    DESCRIPTION = "Combined LucidFlux node - loads model and processes images with optional prompt embeddings"

    def process(self, flux_model, LucidFlux, image, vae, swinir, width, height, CLIP_VISION,
                steps, seed, cfg, use_embeddings=True, prompt_embeddings="none", enable_offload=False,
                positive=None):

        LucidFlux_path = folder_paths.get_full_path("LucidFlux", LucidFlux) if LucidFlux != "none" else None

        if LucidFlux_path is None:
            raise ValueError("LucidFlux checkpoint is required")
        if flux_model is None:
            raise ValueError("Flux model is required - connect from CheckpointLoader")

        is_dev = True
        try:
            if hasattr(flux_model, 'model') and hasattr(flux_model.model, 'model_config'):
                unet_config = flux_model.model.model_config.unet_config
                if hasattr(unet_config, 'get'):
                    is_dev = unet_config.get('guidance_embeds', True)
                elif hasattr(unet_config, 'guidance_embeds'):
                    is_dev = unet_config.guidance_embeds
        except:
            print("Could not auto-detect model type, defaulting to flux-dev")

        model_type = "flux-dev" if is_dev else "flux-schnell"
        print(f"Detected model type: {model_type}")

        cache_key = f"{LucidFlux_path}_{id(flux_model)}_{model_type}"

        if cache_key in self._model_cache:
            print("Using cached LucidFlux model")
            model, state = self._model_cache[cache_key]
        else:
            origin_dict = {
                "name": model_type,
                "offload": enable_offload,
                "device": "cuda:0",
                "output_dir": folder_paths.get_output_directory(),
                "checkpoint": LucidFlux_path,
            }
            args = OmegaConf.create(origin_dict)
            model, state = load_lucidflux_model(args, None, flux_model, device, enable_offload)
            self._model_cache[cache_key] = (model, state)

        prompt_emb_data = None
        if use_embeddings and prompt_embeddings != "none":
            prompt_emb_path = folder_paths.get_full_path("LucidFlux", prompt_embeddings)
            if prompt_emb_path is not None:
                print(f"✓ Loading prompt embeddings from {prompt_emb_path}")
                prompt_emb_data = torch.load(prompt_emb_path, map_location='cpu', weights_only=False)

        print("\n" + "="*60)
        print("🚀 Starting LucidFlux Processing Pipeline")
        print("="*60)
        print_memory_status("📍 [Initial] ")

        print("\n📦 [Encode Start] Preprocessing image...")

        swinir_path = folder_paths.get_full_path("LucidFlux", swinir) if swinir != "none" else None

        if swinir_path is None:
            raise ValueError("SwinIR checkpoint is required")

        SWINIR_MULTIPLE = 64
        adjusted_width = (width // SWINIR_MULTIPLE) * SWINIR_MULTIPLE
        adjusted_height = (height // SWINIR_MULTIPLE) * SWINIR_MULTIPLE

        if adjusted_width < SWINIR_MULTIPLE:
            adjusted_width = SWINIR_MULTIPLE
        if adjusted_height < SWINIR_MULTIPLE:
            adjusted_height = SWINIR_MULTIPLE

        if adjusted_width != width or adjusted_height != height:
            print(f"⚠️ SwinIR dimension adjustment: {width}x{height} → {adjusted_width}x{adjusted_height}")
            print(f"   (Dimensions must be divisible by {SWINIR_MULTIPLE})")

        input_pli_list = tensor2pillist_upscale(image, adjusted_width, adjusted_height)

        # Use embeddings or positive conditioning
        if use_embeddings and prompt_emb_data is not None:
            print("Using precomputed prompt embeddings")
            inp_cond = get_cond_from_embeddings(prompt_emb_data, adjusted_height, adjusted_width, device)
        else:
            print("Using positive conditioning from CLIP")
            if positive is None:
                raise ValueError("Either prompt_embeddings (with use_embeddings=True) or positive conditioning is required")
            inp_cond = get_cond(positive, adjusted_height, adjusted_width, device)

        condition = preprocess_data_cached(
            self._swinir_cache,
            self._redux_cache,
            state,
            swinir_path,
            CLIP_VISION,
            input_pli_list,
            inp_cond,
            device,
            enable_offload
        )

        print("Encoding complete")
        print_memory_status("  After encoding: ")

        # Sampling
        print("\n[Sample Start] Generating restoration...")

        pipe = model.get("model")
        dual_condition_branch = model.get("dual_condition_branch")
        offload = model.get("offload", False)

        print("Pre-denoising cleanup...")
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except:
            pass
        aggressive_cleanup()
        print_memory_status("  After cleanup: ")

        x = lucidflux_inference(pipe, dual_condition_branch, condition, cfg, steps, seed, device,
                               model.get("is_schnell", False), offload)

        # Decoding
        print("\n  [Decode Start] Decoding latents...")
        print_memory_status("  ")

        images = []
        for idx, (i, j) in enumerate(zip(x, condition)):
            print(f"  Decoding image {idx+1}/{len(x)}")

            decoded_image = vae.decode(i).squeeze(0)

            x1 = decoded_image.clamp(-1, 1).to(device)
            hq = wavelet_reconstruction((x1.permute(2, 0, 1) + 1.0) / 2, j.get("ci_pre_origin").squeeze(0).to(device))
            hq = hq.clamp(0, 1)
            hq = hq.unsqueeze(0).permute(0, 2, 3, 1)
            images.append(hq)

            del decoded_image, x1, i
            if idx % 1 == 0:
                aggressive_cleanup()

        img = torch.cat(images, dim=0)

        del x, images, condition
        aggressive_cleanup()

        print_memory_status("✅ [Complete] ")
        print("="*60 + "\n")

        return (img,)


NODE_CLASS_MAPPINGS = {
    "PiePie_Lucidflux": PiePie_Lucidflux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiePie_Lucidflux": "PiePie - Lucidflux",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]