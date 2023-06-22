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
        AbstractModel.__init__(self, "IF")
        self.stageI_model = "XL"
        self.stageII_model = "L"
        self.stages = []

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
        del self.pipe
        gc.collect()
        devices.torch_gc()

    def next_stage(self):
        self.cleanup_on_error()

    def sd_processing_to_dict_encoder(self, p: StableDiffusionProcessing):
        torch.manual_seed(p.seed)
        parameters_dict = {"generator": p.generators, "prompt": p.prompt}

        if p.negative_prompt != "":
            parameters_dict["negative_prompt"] = p.negative_prompt

        return parameters_dict

    def sd_processing_to_dict_generator(self, p: StableDiffusionProcessing):
        generation_parameters = {#"prompt": p.prompt, "negative_prompt": p.negative_prompt,
                                 "prompt_embeds": p.image_embeds, "negative_prompt_embeds": p.negative_image_embeds,
                                 "height": p.height, "width": p.width, "guidance_scale": p.cfg_scale, "num_inference_steps": p.steps}
        return generation_parameters

    def txt2img(self, p, generation_parameters, b):
        if self.current_stage == 1:
            if p.disable_stage_I:
                result_images = [p.init_image for _ in range(p.batch_size)]
            else:
                self.pipe = self.load_pipeline("pipe", IFPipeline, f"DeepFloyd/IF-I-{self.stageI_model}-v1.0", {"safety_checker": None, "watermarker": None})
                result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images

        elif self.current_stage == 2:
            generation_parameters["width"] = p.width2
            generation_parameters["height"] = p.height2
            self.pipe = self.load_pipeline("pipe", IFSuperResolutionPipeline, f"DeepFloyd/IF-II-{self.stageII_model}-v1.0", {"image": p.init_image, "safety_checker": None, "watermarker": None})
            result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        return result_images

    def img2img(self, p, generation_parameters, b):
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        return result_images

    def inpaint(self, p, generation_parameters, b):
        pass
