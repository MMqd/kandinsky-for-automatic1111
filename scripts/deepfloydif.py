from modules import errors
try:
    from diffusers import IFPipeline, IFSuperResolutionPipeline
except ImportError as e:
    errors.print_error_explanation('RESTART AUTOMATIC1111 COMPLETELY TO FINISH INSTALLING PACKAGES FOR kandinsky-for-automatic1111')


import os
import gc
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from packaging import version
from modules import processing, shared, script_callbacks, images, devices, scripts, masking, sd_models, generation_parameters_copypaste, sd_vae#, sd_samplers
from modules.processing import Processed, StableDiffusionProcessing
from modules.shared import opts, state
from modules.sd_models import CheckpointInfo
from modules.paths_internal import script_path
from huggingface_hub import login


import sys
sys.path.append('extensions/kandinsky-for-automatic1111/scripts')
from abstract_model import AbstractModel
#import pdb

class IFModel(AbstractModel):
    pipe = None

    def __init__(self):
        self.stageI_model = "XL"
        self.stageII_model = "L"
        AbstractModel.__init__(self, "IF")

    def load_encoder(self):
        try:
            self.pipe = self.load_pipeline("pipe", IFPipeline, f"DeepFloyd/IF-I-{self.stageI_model}-v1.0", {"safety_checker": None, "watermarker": None})
        except FileNotFoundError as fe:
            errors.print_error_explanation(f'File {fe.filename} not found. Did you forget the Hugging Face token?')

    def run_encoder(self, prior_settings_dict):
        tup = None
        if self.pipe is None:
            errors.print_error_explanation(f'Stage I {self.stageI_model} not loaded. Did you forget the Hugging Face token?')
        elif prior_settings_dict.get("negative_prompt", None) is None:
            tup = self.pipe.encode_prompt(prompt=prior_settings_dict["prompt"])
        else:
            tup = self.pipe.encode_prompt(prompt=prior_settings_dict["prompt"], negative_prompt=prior_settings_dict["negative_prompt"])
        return tup

    def encoder_to_cpu(self):
        pass

    def main_model_to_cpu(self):
        pass

    def cleanup_on_error(self):
        pass

    def txt2img(self, p, generation_parameters, b):
        generation_parameters["prompt_embeds"] = generation_parameters["image_embeds"]
        generation_parameters.pop("image_embeds", None)
        generation_parameters["negative_prompt_embeds"] = generation_parameters["negative_image_embeds"]
        generation_parameters.pop("negative_image_embeds", None)
        generation_parameters.pop("prompt", None)
        generation_parameters.pop("negative_prompt", None)
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        if self.stageII_model != "None":
            self.pipe = self.load_pipeline("pipe", IFSuperResolutionPipeline, f"DeepFloyd/IF-I-{self.stageII_model}-v1.0", {"safety_checker": None, "watermarker": None})
        generation_parameters["width"] = generation_parameters["width"]*4
        generation_parameters["height"] = generation_parameters["height"]*4
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        return result_images

    def img2img(self, p, generation_parameters, b):
        generation_parameters["prompt_embeds"] = generation_parameters["image_embeds"]
        generation_parameters.pop("image_embeds", None)
        generation_parameters["negative_prompt_embeds"] = generation_parameters["negative_image_embeds"]
        generation_parameters.pop("negative_image_embeds", None)
        generation_parameters.pop("prompt", None)
        generation_parameters.pop("negative_prompt", None)
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        return result_images

    def inpaint(self, p, generation_parameters, b):
        pass
