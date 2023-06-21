import sys
import torch
import gradio as gr
from modules import processing, shared, script_callbacks, scripts
from modules.processing import Processed
#import pkg_resources
#import pdb

sys.path.append('extensions/kandinsky-for-automatic1111/scripts')
from deepfloydif import *

def unload_model():
    if shared.sd_model is None:
        shared.sd_model = IFModel()
        print("Unloaded Stable Diffusion model")
        return

    if not isinstance(shared.sd_model, IFModel):
        sd_models.unload_model_weights()
        sd_vae.clear_loaded_vae()
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()
        shared.sd_model = IFModel()

def reload_model():
    if shared.sd_model is None or isinstance(shared.sd_model, IFModel):
        shared.sd_model = None
        sd_models.reload_model_weights()
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()

def unload_if_model():
    if shared.if_model.pipe is not None:
        del shared.if_model.pipe
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()

    if shared.if_model is not None:
        del shared.if_model

    print("Unloaded IF model")

class Script(scripts.Script):
    def title(self):
        return "IF"

    def ui(self, is_img2img):
        gr.Markdown("To save VRAM unload the Stable Diffusion Model")

        unload_sd_model = gr.Button("Unload Stable Diffusion Model")
        unload_sd_model.click(unload_model)
        reload_sd_model = gr.Button("Reload Stable Diffusion Model")
        reload_sd_model.click(reload_model)

        unload_k_model = gr.Button("Unload IF Model")
        unload_k_model.click(unload_if_model)
        with gr.Row():
            unload_sd_model
            reload_sd_model
            unload_k_model

        inputs = []

        return inputs

    def run(self, p) -> Processed:
        p.sampler_name = "DDPM"
        p.init_image = getattr(p, 'init_images', None)
        p.extra_generation_params["Script"] = self.title()

        shared.if_model = getattr(shared, 'if_model', None)

        if shared.if_model is None:
            shared.if_model = IFModel()

        return shared.if_model.process_images(p)
