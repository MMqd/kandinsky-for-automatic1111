from modules import errors
try:
    from diffusers import KandinskyPipeline, KandinskyImg2ImgPipeline, KandinskyPriorPipeline, KandinskyInpaintPipeline, KandinskyV22Pipeline, KandinskyV22PriorPipeline
    from diffusers.models import UNet2DConditionModel
    from transformers import CLIPVisionModelWithProjection
except ImportError as e:
    errors.print_error_explanation('RESTART AUTOMATIC1111 COMPLETELY TO FINISH INSTALLING PACKAGES FOR kandinsky-for-automatic1111')

import os
import gc
import torch
from PIL import Image
from modules import processing, shared, script_callbacks, images, devices, scripts, masking, sd_models, generation_parameters_copypaste, sd_vae#, sd_samplers
from modules.processing import Processed, StableDiffusionProcessing

import sys
sys.path.append('extensions/kandinsky-for-automatic1111/scripts')
from abstract_model import AbstractModel
#import pdb

move_to_cuda=False

class KandinskyModel(AbstractModel):
    def __init__(self, cache_dir="", version="2.1"):
        AbstractModel.__init__(self, cache_dir="Kandinsky", version=version)
        self.image_encoder = None
        self.pipe_prior = None
        self.pipe = None
        self.unet = None
        self.low_vram = True

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
                        num_inference_steps=p.prior_inference_steps,
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
                self.pipe = self.load_pipeline("pipe", KandinskyPipeline, f"kandinsky-community/kandinsky-{self.version}".replace(".", "-"))

                result_images[i] = self.pipe(**generation_parameters, num_images_per_prompt=1).images[0]
                self.pipe.to("cpu")
        return result_images

    def load_encoder(self):
        if self.version == "2.1":
            if self.pipe_prior is None:
                self.pipe_prior = self.load_pipeline("pipe_prior", KandinskyPriorPipeline, f"kandinsky-community/kandinsky-{self.version}-prior".replace(".", "-"))
        elif self.version == "2.2":
            if self.image_encoder is None:
                self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    'kandinsky-community/kandinsky-2-2-prior',
                    subfolder='image_encoder',
                    cache_dir=os.path.join(self.models_path, "kandinsky22"),
                    low_cpu_mem_usage=True
                    # local_files_only=True
                )
                
                self.image_encoder.to("cpu" if self.low_vram else "cuda")

                self.pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
                    'kandinsky-community/kandinsky-2-2-prior',
                    image_encoder=self.image_encoder,
                    torch_dtype=torch.float32,
                    cache_dir=os.path.join(self.models_path, "kandinsky22"),
                    low_cpu_mem_usage=True
                    # local_files_only=True
                )

                self.image_encoder.to("cpu" if self.low_vram else "cuda")

                self.unet = UNet2DConditionModel.from_pretrained(
                    'kandinsky-community/kandinsky-2-2-decoder',
                    subfolder='unet',
                    cache_dir=os.path.join(self.models_path, "kandinsky22"),
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                    # local_files_only=True
                ).half().to("cuda")

                self.pipe = KandinskyV22Pipeline.from_pretrained(
                    'kandinsky-community/kandinsky-2-2-decoder',
                    unet=self.unet,
                    torch_dtype=torch.float16,
                    cache_dir=os.path.join(self.models_path, "kandinsky22"),
                    low_cpu_mem_usage=True
                    # local_files_only=True
                ).to("cuda")

    def run_encoder(self, prior_settings_dict):
        self.main_model_to_cpu()
        return self.pipe_prior(**prior_settings_dict).to_tuple()

    def encoder_to_cpu(self):
        pass
        #self.image_encoder.to("cpu")
        #self.pipe_prior.to("cpu")
        #self.pipe.to("cuda")
        #self.unet.to("cuda")

    def unload(self):
        if self.image_encoder is not None:
            self.image_encoder.to("cpu")
            del self.image_encoder

        if self.pipe_prior is not None:
            self.pipe_prior.to("cpu")
            del self.pipe_prior

        if self.pipe is not None:
            self.pipe.to("cpu")
            del self.pipe

        if self.unet is not None:
            self.unet.to("cpu")
            del self.unet
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()

    def main_model_to_cpu(self):
        pass
        #self.pipe.to("cpu")
        #self.unet.to("cpu")
        #self.image_encoder.to("cuda")
        #self.pipe_prior.to("cuda")

    def sd_processing_to_dict_encoder(self, p: StableDiffusionProcessing):
        torch.manual_seed(0)
        parameters_dict = {"generator": p.generators, "prompt": p.prompt}
        parameters_dict["guidance_scale"] = p.prior_cfg_scale#getattr(p, "prior_cfg_scale", 4)
        parameters_dict["num_inference_steps"] = p.prior_inference_steps#getattr(p, "prior_inference_steps", 20)

        if p.negative_prompt != "":
            parameters_dict["negative_prompt"] = p.negative_prompt

        return parameters_dict

    def sd_processing_to_dict_generator(self, p: StableDiffusionProcessing):
        if self.version == "2.1":
            generation_parameters = {"prompt": p.prompt, "negative_prompt": p.negative_prompt, "image_embeds": p.image_embeds, "negative_image_embeds": p.negative_image_embeds,
                                    "height": p.height, "width": p.width, "guidance_scale": p.cfg_scale, "num_inference_steps": p.steps}
        elif self.version == "2.2":
            generation_parameters = {"image_embeds": p.image_embeds.half(), "negative_image_embeds": p.negative_image_embeds.half(),
                                    "height": p.height, "width": p.width, "guidance_scale": p.cfg_scale, "num_inference_steps": p.steps}
        return generation_parameters


    def cleanup_on_error(self):
        if self.pipe_prior is not None:
            self.main_model_to_cpu()

        if self.pipe is not None:
            self.encoder_to_cpu()

    def txt2img(self, p, generation_parameters, b):
        if self.version == "2.1":
            self.pipe = self.load_pipeline("pipe", KandinskyPipeline, f"kandinsky-community/kandinsky-{self.version}".replace(".", "-"), move_to_cuda=move_to_cuda)
        #else:
        #    self.unet.to("cuda")
        #    self.pipe.to("cuda")

        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images
        return self.mix_images(p, generation_parameters, b, result_images)

    def img2img(self, p, generation_parameters, b):
        self.pipe = self.load_pipeline("pipe", KandinskyImg2ImgPipeline, f"kandinsky-community/kandinsky-{self.version}".replace(".", "-"), move_to_cuda=move_to_cuda)
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size, image=p.init_image, strength=p.denoising_strength).images
        return self.mix_images(p, generation_parameters, b, result_images)

    def inpaint(self, p, generation_parameters, b):
        self.pipe = self.load_pipeline("pipe", KandinskyInpaintPipeline, f"kandinsky-community/kandinsky-{self.version}-inpaint".replace(".", "-"), move_to_cuda=move_to_cuda)
        result_images = self.pipe(**generation_parameters, num_images_per_prompt=p.batch_size, image=p.new_init_image, mask_image=p.new_mask).images
        return self.mix_images(p, generation_parameters, b, result_images)
