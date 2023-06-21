from modules import errors
try:
    from diffusers import IFPipeline
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

import sys
sys.path.append('extensions/kandinsky-for-automatic1111/scripts')
from abstract_model import AbstractModel
#import pdb

class IFModel(AbstractModel):
    pipe = None

    def __init__(self):
        self.cache_dir = os.path.join(os.path.join(script_path, 'models'), 'IF')

    def load_encoder(self):
        pass

    def run_encoder(self, prior_settings_dict):
        if prior_settings_dict.get("negative_prompt", None) is None:
            tup = self.pipe.encode_prompt(prompt=prior_settings_dict["prompt"])
        else:
            tup = self.pipe.encode_prompt(prompt=prior_settings_dict["prompt"], negative_prompt=prior_settings_dict["negative_prompt"])
        return tup.to_tuple()

    def encoder_to_cpu(self):
        pass

    def main_model_to_cpu(self):
        pass

    def cleanup_on_error(self):
        pass

    def txt2img(self, p, generation_parameters, b):
        self.pipe = self.load_pipeline("pipe", IFPipeline, "DeepFloyd/IF-I-XL-v1.0")
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        return result_images

    def img2img(self, p, generation_parameters, b):
        pass

    def inpaint(self, p, generation_parameters, b):
        pass
