from modules import errors
try:
    from diffusers import DiffusionPipeline
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
#import pdb

class KProcessed(processing.Processed):
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = int(seed)
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = comments
        self.width = p.width
        self.height = p.height
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None

        self.sd_model_hash = p.sd_model_hash

        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = state.job_timestamp
        self.clip_skip = 1
        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if not isinstance(self.prompt, list) else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if not isinstance(self.seed, list) else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

class KandinskyCheckpointInfo(CheckpointInfo):
    def __init__(self, filename="kandinsky21"):
        self.filename = filename
        name = "kandinsky21"
        self.name = name
        self.name_for_extra = "kandinsky21_extra"#os.path.splitext(os.path.basename(filename))[0]
        self.model_name = "kandinsky21"#os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = "0000000000000000000000000000000000000000000000000000000000000000"#model_hash(filename)
        self.sha256 = "0000000000000000000000000000000000000000000000000000000000000000"#hashes.sha256_from_cache(self.filename, "checkpoint/" + name)
        self.shorthash = self.sha256[0:10] if self.sha256 else None
        self.sd_model_hash = self.shorthash
        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.ids = [self.hash, self.model_name, self.title, name, f'{name} [{self.hash}]'] + ([self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])
        self.metadata = {}

    def register(self):
        return
    #checkpoints_list[self.title] = self
    #    for i in self.ids:
    #        checkpoint_aliases[i] = self

    def calculate_shorthash(self):
        self.sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
        #if self.sha256 is None:
        #    return
        self.shorthash = self.sha256[0:10]
        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']
        #checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        #self.register()
        return self.shorthash

def truncate_string(string, max_length=images.max_filename_part_length, encoding='utf-8'):
    return string.encode(encoding)[:max_length].decode(encoding, 'ignore')

class AbstractModel():
    attention_type = 'auto'#'max'
    cache_dir = os.path.join(os.path.join(script_path, 'models'), "Kandinsky")
    cond_stage_key = "edit"
    sd_checkpoint_info = KandinskyCheckpointInfo()
    sd_model_hash = sd_checkpoint_info.shorthash
    cached_image_embeds = {"settings": {}, "embeds": (None, None)}

    def load_pipeline(self, pipe_name: str, pipeline: DiffusionPipeline, pretrained_model_name_or_path):
        pipe = getattr(self, pipe_name, None)

        if not isinstance(pipe, pipeline) or pipe is None:
            if pipe is not None:
                pipe = None
                gc.collect()
                devices.torch_gc()
            pipe = pipeline.from_pretrained(pretrained_model_name_or_path, variant="fp16", torch_dtype=torch.float16, cache_dir=self.cache_dir)#, scheduler=dpm)
            pipe.to("cuda")
            #pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing(self.attention_type)
            #pipe.unet.to(memory_format=torch.channels_last)
            setattr(self, pipe_name, pipe)
        else:
            pipe.to("cuda")

        return pipe

    def create_infotext(self, p: StableDiffusionProcessing, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0): # pylint: disable=unused-argument
        index = position_in_batch + iteration * p.batch_size

        generation_params = {
            "Steps": p.steps,
            "Sampler": p.sampler_name,
            "CFG scale": p.cfg_scale,
            "Image CFG scale": getattr(p, 'image_cfg_scale', None),
            "Seed": all_seeds[index],
            "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
            "Size": f"{p.width}x{p.height}",
            "Model hash": None,
            "Model": None,
            "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
            "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
            "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
            "Denoising strength": getattr(p, 'denoising_strength', None),
            "Conditional mask weight": getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
            "ENSD": None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta,
            "Init image hash": getattr(p, 'init_img_hash', None)
        }
        generation_params.update(p.extra_generation_params)

        generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

        negative_prompt_text = "\nNegative prompt: " + p.all_negative_prompts[index] if p.all_negative_prompts[index] else ""

        return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip()

    def load_encoder(self ):
        raise NotImplementedError("load_encoder method not implemented")

    def run_encoder(self, prior_settings_dict):
        raise NotImplementedError("run_encoder method not implemented")

    def txt2img(self, p, generation_parameters, b):
        raise NotImplementedError("txt2img method not implemented")

    def img2img(self, p, generation_parameters, b):
        raise NotImplementedError("img2img method not implemented")

    def inpaint(self, p, generation_parameters, b):
        raise NotImplementedError("inpaint method not implemented")

    def encoder_to_cpu(self):
        raise NotImplementedError("encoder_to_cpu method not implemented")

    def main_model_to_cpu(self):
        raise NotImplementedError("main_model_to_cpu method not implemented")

    def cleanup_on_error(self):
        raise NotImplementedError("cleanup_on_error method not implemented")

    def process_images(self, p: StableDiffusionProcessing) -> Processed:
        try:
            state.begin()
            processing.fix_seed(p)
            devices.torch_gc()
            gc.collect()
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False

            all_result_images = []
            initial_infos = []
            p.all_negative_prompts = [p.negative_prompt] * p.n_iter * p.batch_size
            p.all_prompts = [p.prompt] * p.n_iter * p.batch_size
            p.seed = int(p.seed)
            p.all_seeds = [p.seed + j for j in range(p.n_iter * p.batch_size)]
            initial_info = self.create_infotext(p, p.all_prompts, p.all_seeds, p.all_seeds, iteration=0, position_in_batch=0)

            p.sd_model_hash = self.sd_checkpoint_info.sd_model_hash

            if version.parse(torch.version.cuda) < version.parse("10.2"):
                torch.use_deterministic_algorithms(True)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            torch.manual_seed(0)

            if p.init_image is not None:
                p.init_image = p.init_image[0]
                p.init_image = images.flatten(p.init_image, opts.img2img_background_color)

            state.job = "Prior"
            print("Starting Prior")
            if p.batch_size * p.n_iter > 1:
                generators = []
                for i in range(p.batch_size * p.n_iter):
                    generators.append(torch.Generator().manual_seed(p.seed + i))
            else:
                generators = torch.Generator().manual_seed(p.seed)


            prior_settings_dict = {"generator": generators, "prompt": p.prompt, "guidance_scale": p.prior_cfg_scale}
            prior_settings_dict["num_inference_steps"] = p.inference_steps

            if p.negative_prompt != "":
                prior_settings_dict["negative_prompt"] = p.negative_prompt

            if self.cached_image_embeds["settings"] == prior_settings_dict:
                image_embeds, negative_image_embeds = self.cached_image_embeds["embeds"]
            else:
                self.load_encoder()

                if state.interrupted:
                    self.encoder_to_cpu()
                    gc.collect()
                    devices.torch_gc()
                    torch.cuda.empty_cache()
                    return KProcessed(p, [], p.seed, initial_info, all_seeds=p.all_seeds)

                image_embeds, negative_image_embeds = self.run_encoder(prior_settings_dict)

                self.cached_image_embeds["settings"] = prior_settings_dict
                self.cached_image_embeds["embeds"] = (image_embeds, negative_image_embeds)

            self.encoder_to_cpu()
            devices.torch_gc()
            gc.collect()
            torch.cuda.empty_cache()

            print("Finished Prior")

            generation_parameters = {"prompt": p.prompt, "negative_prompt": p.negative_prompt, "image_embeds": image_embeds, "negative_image_embeds": negative_image_embeds,
                                    "height": p.height, "width": p.width, "guidance_scale": p.cfg_scale, "num_inference_steps": p.steps}

            state.job_no = p.n_iter * p.batch_size

            for b in range(p.n_iter):
                if state.interrupted:
                    break

                for batchid in range(p.batch_size):
                    initial_infos.append(self.create_infotext(p, p.all_prompts, p.all_seeds, p.all_seeds, iteration=b, position_in_batch=batchid))

                state.job = "Generating"
                if p.init_image is None:
                    result_images = self.txt2img(p, generation_parameters, b)

                else:
                    if p.image_mask is None:
                        if p.denoising_strength != 0:
                            result_images = self.img2img(p, generation_parameters, b)
                        else:
                            result_images = [p.init_image] * p.batch_size

                    else:
                        crop_region = None
                        if not p.inpainting_mask_invert:
                            p.image_mask = ImageOps.invert(p.image_mask)

                        if p.mask_blur > 0:
                            p.image_mask = p.image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

                        mask = p.image_mask
                        mask = mask.convert('L')
                        new_init_image = p.init_image

                        if p.inpaint_full_res:
                            mask = ImageOps.invert(mask)
                            crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
                            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)
                            x1, y1, x2, y2 = crop_region
                            mask = mask.crop(crop_region)
                            mask = images.resize_image(2, mask, p.width, p.height)
                            p.paste_to = (x1, y1, x2-x1, y2-y1)

                            new_init_image = new_init_image.crop(crop_region)
                            new_init_image = images.resize_image(2, new_init_image, p.width, p.height)
                            mask = ImageOps.invert(mask)
                        else:
                            p.image_mask = images.resize_image(p.resize_mode, p.image_mask, p.width, p.height)
                            np_mask = np.array(p.image_mask)
                            np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                            mask = Image.fromarray(np_mask)

                        p.new_init_image = new_init_image
                        p.new_mask = mask
                        result_images = self.inpaint(p, generation_parameters, b)

                        if p.inpaint_full_res:
                            for i in range(len(result_images)):
                                paste_loc = p.paste_to
                                x, y, w, h = paste_loc
                                base_image = Image.new('RGBA', (p.init_image.width, p.init_image.height))
                                mask = ImageOps.invert(mask)
                                result_images[i] = images.resize_image(1, result_images[i], w, h)
                                mask = images.resize_image(1, mask, w, h)
                                mask = mask.convert('L')

                                base_image.paste(result_images[i], (x, y), mask=mask)
                                image = p.init_image
                                image = image.convert('RGBA')
                                image.alpha_composite(base_image)
                                image.convert('RGB')
                                processing.apply_color_correction(processing.setup_color_correction(p.init_image), image)
                                result_images[i] = image
                        #else:
                        #    for i in range(len(result_images)):
                        #        base_image = result_images[i]
                        #        base_image = base_image.convert('RGBA')
                        #        mask = ImageOps.invert(mask)
                        #        mask = mask.convert('L')
                        #        base_image.putalpha(mask)

                        #        image = images.resize_image(1, init_image, p.width, p.height)
                        #        image = image.convert('RGBA')
                        #        image.alpha_composite(base_image)
                        #        image.convert('RGB')
                        #        processing.apply_color_correction(processing.setup_color_correction(init_image), image)
                        #        result_images[i] = image

                for imgid, result_image in enumerate(result_images):
                    images.save_image(result_image, p.outpath_samples, "", p.all_seeds[imgid],
                                      truncate_string(p.prompt[0] if isinstance(p.prompt, list) else p.prompt),
                                      opts.samples_format, info=initial_infos[imgid], p=p)

                all_result_images.extend(result_images)

                state.job_no = b * p.batch_size
                state.current_image = result_images[0]
                state.nextjob()

            self.main_model_to_cpu()

            #del pipe
            del generators
            gc.collect()
            devices.torch_gc()
            torch.cuda.empty_cache()

            output_images = all_result_images

            # Save Grid
            unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
            if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
                grid = images.image_grid(output_images, p.batch_size)

                if opts.return_grid:
                    text = initial_info
                    if opts.enable_pnginfo:
                        grid.info["parameters"] = text
                    output_images.insert(0, grid)

                if opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], truncate_string(p.all_prompts[0]), opts.grid_format, info=initial_info, short_filename=not opts.grid_extended_filename, p=p, grid=True)


            p.n_iter = 1
            state.end()

            return KProcessed(p, all_result_images, p.seed, initial_info, all_seeds=p.all_seeds)

        except RuntimeError as re:
            self.cleanup_on_error()

            gc.collect()
            devices.torch_gc()
            torch.cuda.empty_cache()
            if str(re).startswith('CUDA out of memory.'):
                print("OutOfMemoryError: CUDA out of memory.")
            return
