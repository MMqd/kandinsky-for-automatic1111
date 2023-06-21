from modules import errors
try:
    from diffusers import KandinskyPipeline, KandinskyImg2ImgPipeline, KandinskyPriorPipeline, KandinskyInpaintPipeline
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

class KandinskyModel(AbstractModel):
    pipe = None
    pipe_prior = None

    def mix_images(self, p, generation_parameters, b, result_images):
        if p.extra_image != [] and p.extra_image is not None:
            img2 = Image.fromarray(p.extra_image)
            for i in range(len(result_images)):
                img1 = result_images[i]
                generation_parameters = dict(generation_parameters)
                self.pipe.to("cpu")
                self.pipe_prior.to("cuda")
                image_embeds, negative_image_embeds = self.pipe_prior.interpolate(
                        [img1, img2],
                        [p.img1_strength, p.img2_strength],
                        #negative_prior_prompt=p.negative_prompt,
                        num_inference_steps=p.inference_steps,
                        num_images_per_prompt = 1,
                        #num_images_per_prompt = p.batch_size,
                        generator=torch.Generator().manual_seed(p.seed + i + b * p.batch_size),
                        guidance_scale=p.prior_cfg_scale).to_tuple()
                generation_parameters["image_embeds"] = image_embeds
                generation_parameters["negative_image_embeds"] = negative_image_embeds

                p.extra_generation_params["Image 1 Strength"] = p.img1_strength
                p.extra_generation_params["Image 2 Strength"] = p.img2_strength
                p.extra_generation_params["Extra Image"] = ""#self.img2_name

                self.pipe_prior.to("cpu")
                self.pipe = self.load_pipeline("pipe", KandinskyPipeline, "kandinsky-community/kandinsky-2-1")

                result_images[i] = self.pipe(**generation_parameters, num_images_per_prompt=1).images[0]
                self.pipe.to("cpu")
        return result_images

    def load_encoder(self):
        self.pipe_prior = self.load_pipeline("pipe_prior", KandinskyPriorPipeline, "kandinsky-community/kandinsky-2-1-prior")

    def run_encoder(self, prior_settings_dict):
        return self.pipe_prior(**prior_settings_dict).to_tuple()

    def encoder_to_cpu(self):
        self.pipe_prior.to("cpu")

    def main_model_to_cpu(self):
        self.pipe.to("cpu")

    def cleanup_on_error(self):
        if self.pipe_prior is not None:
            self.main_model_to_cpu()

        if self.pipe is not None:
            self.encoder_to_cpu()

    def txt2img(self, p, generation_parameters, b):
        self.pipe = self.load_pipeline("pipe", KandinskyPipeline, "kandinsky-community/kandinsky-2-1")
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        return self.mix_images(p, generation_parameters, b, result_images)

    def img2img(self, p, generation_parameters, b):
        self.pipe = self.load_pipeline("pipe", KandinskyImg2ImgPipeline, "kandinsky-community/kandinsky-2-1")
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size, image=p.init_image, strength=p.denoising_strength).images
        return self.mix_images(p, generation_parameters, b, result_images)

    def inpaint(self, p, generation_parameters, b):
        self.pipe = self.load_pipeline("pipe", KandinskyInpaintPipeline, "kandinsky-community/kandinsky-2-1-inpaint")
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size, image=p.new_init_image, mask_image=p.new_mask).images
        return self.mix_images(p, generation_parameters, b, result_images)
