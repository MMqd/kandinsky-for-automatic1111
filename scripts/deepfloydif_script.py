from modules import errors
try:
    from huggingface_hub import login
except ImportError as e:
    errors.print_error_explanation('RESTART AUTOMATIC1111 COMPLETELY TO FINISH INSTALLING PACKAGES FOR kandinsky-for-automatic1111')


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
        unload_sd_model = None
        reload_sd_model = None
        unload_k_model = None

        with gr.Row():
            unload_sd_model = gr.Button("Unload Stable Diffusion Model")
            unload_sd_model.click(unload_model)
            reload_sd_model = gr.Button("Reload Stable Diffusion Model")
            reload_sd_model.click(reload_model)
            unload_k_model = gr.Button("Unload IF Model")
            unload_k_model.click(unload_if_model)

        stageI_model = None
        stageII_model = None
        with gr.Row():
            stageI_model = gr.inputs.Dropdown(label="Stage I Model Type", choices=["None", "M", "L", "XL"], default="XL")
            stageII_model = gr.inputs.Dropdown(label="Stage II Model Type", choices=["None", "M", "L"], default="L")

        token_textbox = gr.inputs.Textbox(label="Hugging Face Token", type="password")

        inputs = [token_textbox, stageI_model, stageII_model]

        return inputs

    def run(self, p, token, stageI_model, stageII_model) -> Processed:
        p.sampler_name = "DDPM"
        p.init_image = getattr(p, 'init_images', None)
        p.extra_generation_params["Script"] = "if"

        shared.if_model = getattr(shared, 'if_model', None)

        if shared.if_model is None:
            shared.if_model = IFModel()
            shared.if_model.stageI_model = stageI_model
            shared.if_model.stageII_model = stageII_model
            if token != "":
                login(token=token)

        return shared.if_model.process_images(p)
