import sys
import torch
import gradio as gr
from modules import processing, shared, script_callbacks, scripts
from modules.processing import Processed
#import pkg_resources
#import pdb

sys.path.append('extensions/kandinsky-for-automatic1111/scripts')
from kandinsky import *

def unload_model():
    if shared.sd_model is None:
        shared.sd_model = KandinskyModel()
        print("Unloaded Stable Diffusion model")
        return

    if not isinstance(shared.sd_model, KandinskyModel):
        sd_models.unload_model_weights()
        sd_vae.clear_loaded_vae()
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()
        shared.sd_model = KandinskyModel()

def reload_model():
    if shared.sd_model is None or isinstance(shared.sd_model, KandinskyModel):
        shared.sd_model = None
        sd_models.reload_model_weights()
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()

def unload_kandinsky_model():
    if getattr(shared, "kandinsky_model", None) is not None:
        shared.kandinsky_model.unload()

        del shared.kandinsky_model
        print("Unloaded Kandinsky model")

    else:
        print("No Kandinsky model to unload")


class Script(scripts.Script):
    def title(self):
        return "Kandinsky"

    def ui(self, is_img2img):
        gr.Markdown("To save VRAM unload the Stable Diffusion Model")

        unload_sd_model = None
        reload_sd_model = None
        unload_k_model = None

        prior_inference_steps = None
        prior_cfg_scale = None

        with gr.Row():
            unload_sd_model = gr.Button("Unload Stable Diffusion Model")
            unload_sd_model.click(unload_model)
            reload_sd_model = gr.Button("Reload Stable Diffusion Model")
            reload_sd_model.click(reload_model)
            unload_k_model = gr.Button("Unload Kandinsky Model")
            unload_k_model.click(unload_kandinsky_model)

        with gr.Row():
            prior_inference_steps = gr.inputs.Slider(minimum=2, maximum=1024, step=1, label="Prior Inference Steps", default=64)
            prior_cfg_scale = gr.inputs.Slider(minimum=1, maximum=20, step=0.5, label="Prior CFG Scale", default=4)

        model_version = gr.inputs.Dropdown(["2.1", "2.2"], label="Kandinsky Version", default="2.2")
        gr.Markdown("Kandinsky 2.2 requires much more RAM, and only txt2img is currently supported")
        low_vram = gr.inputs.Checkbox(label="Kandinsky 2.2 Low VRAM", default=True)

        with gr.Accordion("Image Mixing", open=False):
            with gr.Row():
                img1_strength = gr.inputs.Slider(minimum=-2, maximum=2, label="Interpolate Image 1 Strength", default=0.5)
                img2_strength = gr.inputs.Slider(minimum=-2, maximum=2, label="Interpolate Image 2 Strength (image below)", default=0.5)
            extra_image = gr.inputs.Image()

        inputs = [extra_image, prior_inference_steps, prior_cfg_scale, model_version, img1_strength, img2_strength, low_vram]

        return inputs

    def run(self, p, extra_image, prior_inference_steps, prior_cfg_scale, model_version, img1_strength, img2_strength, low_vram) -> Processed:
        p.extra_image = extra_image
        p.prior_inference_steps = prior_inference_steps
        p.prior_cfg_scale = prior_cfg_scale
        p.img1_strength = img1_strength
        p.img2_strength = img2_strength
        if model_version == "2.1":
            p.sampler_name = "DDIM"
        elif model_version == "2.2":
            p.sampler_name = "DDPM"
        p.init_image = getattr(p, 'init_images', None)
        p.extra_generation_params["Prior Inference Steps"] = prior_inference_steps
        p.extra_generation_params["Prior CFG Scale"] = prior_cfg_scale
        p.extra_generation_params["Script"] = self.title()
        p.extra_generation_params["Kandinsky Version"] = model_version

        shared.kandinsky_model = getattr(shared, 'kandinsky_model', None)

        if shared.kandinsky_model is None or shared.kandinsky_model.version != model_version or (model_version == "2.2" and shared.kandinsky_model.low_vram != low_vram):
            shared.kandinsky_model = KandinskyModel(version=model_version)
            shared.kandinsky_model.low_vram = low_vram

        return shared.kandinsky_model.process_images(p)
