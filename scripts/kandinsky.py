import torch, os
from diffusers import DiffusionPipeline, KandinskyPipeline, KandinskyImg2ImgPipeline, KandinskyPriorPipeline, KandinskyInpaintPipeline, DPMSolverMultistepScheduler
import gradio as gr
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from modules import processing, shared, script_callbacks, images, devices, scripts, masking, sd_models, generation_parameters_copypaste, sd_vae#, sd_samplers
from modules.processing import Processed, StableDiffusionProcessing
from modules.shared import opts, state
from modules.sd_models import CheckpointInfo
import gc
from packaging import version
from modules.paths_internal import script_path
#import pkg_resources
#import pdb

class KandinskyModel():
    cond_stage_key = "edit"
    sd_checkpoint_info = KandinskyCheckpointInfo()

class KandinskyCheckpointInfo(CheckpointInfo):
    def __init__(self, filename="kandinsky21"):
        self.filename = filename
        abspath = os.path.join(os.path.join(script_path, 'models'), "Kandinsky")
        #if shared.opts.ckpt_dir is not None and abspath.startswith(shared.opts.ckpt_dir):
        #    name = abspath.replace(shared.opts.ckpt_dir, '')
        #elif abspath.startswith(model_path):
        #    name = abspath.replace(model_path, '')
        #else:
        #    name = os.path.basename(filename)
        #if name.startswith("\\") or name.startswith("/"):
        #    name = name[1:]
        self.name = "kandinsky21"
        self.name_for_extra = "kandinsky21_extra"#os.path.splitext(os.path.basename(filename))[0]
        self.model_name = "kandinsky21"#os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = "0000000000000000000000000000000000000000000000000000000000000000"#model_hash(filename)
        self.sha256 = "0000000000000000000000000000000000000000000000000000000000000000"#hashes.sha256_from_cache(self.filename, "checkpoint/" + name)
        self.shorthash = self.sha256[0:10] if self.sha256 else None
        self.title = if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.ids = [self.hash, self.model_name, self.title, name, f'{name} [{self.hash}]'] + ([self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])
        self.metadata = {}
        #_, ext = #os.path.splitext(self.filename)
        #if ext.lower() == ".safetensors":
        #    try:
        #        self.metadata = read_metadata_from_safetensors(filename)
        #    except Exception as e:
        #        errors.display(e, f"reading checkpoint metadata: {filename}")

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


def unload_model():
    #print(type(shared.sd_vae))
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
        #sd_vae.clear_loaded_vae()


def reload_model():
    if shared.sd_model is None or isinstance(shared.sd_model, KandinskyModel):
        shared.sd_model = None
        sd_models.reload_model_weights()
        #sd_vae.reload_vae_weights()
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()

def unload_kandinsky_model():
    pipe_prior = getattr(shared, 'pipe_prior', None)

    if pipe_prior != None:
        del shared.pipe_prior
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()

    pipe = getattr(shared, 'pipe', None)

    if pipe != None:
        del shared.pipe
        devices.torch_gc()
        gc.collect()
        torch.cuda.empty_cache()
    print("Unloaded Kandinsky model")

class Script(scripts.Script):
    attention_type = 'auto'#'max'
    cache_dir="models/Kandinsky"
    #img2_name = ""

    def title(self):
        return "Kandinsky"

    def ui(self, is_img2img):
        model_loading_help = gr.Markdown("To save vram unload the Stable Diffusion Model")

        unload_sd_model = gr.Button("Unload Stable Diffusion Model")
        unload_sd_model.click(unload_model)
        reload_sd_model = gr.Button("Reload Stable Diffusion Model")
        reload_sd_model.click(reload_model)

        unload_k_model = gr.Button("Unload Kandinsky Model")
        unload_k_model.click(unload_kandinsky_model)
        with gr.Row():
            unload_sd_model
            reload_sd_model
            unload_k_model
        inference_steps = gr.inputs.Slider(minimum=2, maximum=1024, step=1, label="Prior Inference Steps", default=128)
        prior_cfg_scale = gr.inputs.Slider(minimum=1, maximum=20, step=0.5, label="Prior CFG SCale", default=4)

        with gr.Accordion("Image Mixing", open=False):
            img1_strength = gr.inputs.Slider(minimum=-2, maximum=2, label="Interpolate Image 1 Strength", default=0.5)
            img2_strength = gr.inputs.Slider(minimum=-2, maximum=2, label="Interpolate Image 2 Strength (image below)", default=0.5)
            extra_image = gr.inputs.Image()

        return [extra_image, inference_steps, prior_cfg_scale, img1_strength, img2_strength]#, img2_name]
    
    def mix_images(self, p, pipe, pipe_prior, generation_parameters, img1_strength, img2_strength, inference_steps, prior_cfg_scale, img1, img2, generators):#, img2_name
        generation_parameters = dict(generation_parameters)
        pipe.to("cpu")
        pipe_prior.to("cuda")
        image_embeds, negative_image_embeds = pipe_prior.interpolate(
                [img1, img2],
                [img1_strength, img2_strength],
                #negative_prior_prompt=p.negative_prompt,
                num_inference_steps=inference_steps,
                num_images_per_prompt = 1,
                #num_images_per_prompt = p.batch_size,
                generator=generators,
                guidance_scale=prior_cfg_scale).to_tuple()
        generation_parameters["image_embeds"] = image_embeds
        generation_parameters["negative_image_embeds"] = negative_image_embeds

        p.extra_generation_params["Image 1 Strength"] = img1_strength
        p.extra_generation_params["Image 2 Strength"] = img2_strength
        p.extra_generation_params["Extra Image"] = ""#self.img2_name

        pipe_prior.to("cpu")
        #pdb.set_trace()

        if not isinstance(pipe, KandinskyPipeline) or pipe == None:
            pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", variant="fp16", torch_dtype=torch.float16, cache_dir=cache_dir)#, scheduler=dpm)
            pipe.to("cuda")
            #pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing(self.attention_type)
            pipe.unet.to(memory_format=torch.channels_last)
            shared.pipe = pipe
        else:
            pipe.to("cuda")

        result = pipe(**generation_parameters, num_images_per_prompt=1).images[0]
        pipe.to("cpu")
        return result

    def run(self, p, extra_image, inference_steps, prior_cfg_scale, img1_strength, img2_strength):#, img2_name):
        try:
            #img2_name = extra_image.name
            #unload_model()
      #      if not isinstance(shared.sd_model, str):
      #          sd_models.unload_model_weights()
      #          sd_vae.clear_loaded_vae()
      #  #        shared.sd_model = "kandinsky"

            state.begin()
            processing.fix_seed(p)
            devices.torch_gc()
            gc.collect()
            torch.cuda.empty_cache()
            #shared.opts.sd_model_checkpoint = None

            #os.environ["TRANSFORMERS_CACHE"] = "models/Kandinsky"

            torch.backends.cudnn.benchmark = False
            #print(torch.version.cuda)

            #try:

            if version.parse(torch.version.cuda) < version.parse("10.2"):
                torch.use_deterministic_algorithms(True)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            #except pkg_resources.DistributionNotFound:
            #    print(f'CUDA not found')
            #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            torch.manual_seed(0)

            init_image = getattr(p, 'init_images', None)
            if init_image != None:
                init_image = init_image[0]
                init_image = images.flatten(init_image, opts.img2img_background_color)

            state.job = "Prior"
            print("Starting Prior")

            pipe_prior = getattr(shared, 'pipe_prior', None)

            if pipe_prior == None:
                pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, cache_dir=self.cache_dir)
                pipe_prior.to("cuda")
                #pipe_prior.enable_sequential_cpu_offload()
                pipe_prior.enable_attention_slicing(self.attention_type)
                shared.pipe_prior = pipe_prior
            else:
                pipe_prior.to("cuda")

            seed = int(p.seed)
            all_seeds = [seed + j for j in range(p.n_iter * p.batch_size)]

            if p.batch_size * p.n_iter > 1:
                generators = []
                for i in range(p.batch_size * p.n_iter):
                    generators.append(torch.Generator().manual_seed(seed + i))
            else:
                generators = torch.Generator().manual_seed(seed)

            p.extra_generation_params["Inference Steps"] = inference_steps
            p.extra_generation_params["Model"] = "kandinsky21"
            p.extra_generation_params["Model hash"] = "0000000000"
            p.extra_generation_params["Prior CFG Scale"] = prior_cfg_scale

            prior_settings_dict = {"generator": generators, "prompt": p.prompt, "guidance_scale": prior_cfg_scale}
            prior_settings_dict["num_inference_steps"] = inference_steps

            if p.negative_prompt != "":
                prior_settings_dict["negative_prompt"] = p.negative_prompt

            image_embeds, negative_image_embeds = pipe_prior(**prior_settings_dict).to_tuple()

            #del pipe_prior
            pipe_prior.to("cpu")
            devices.torch_gc()
            gc.collect()
            torch.cuda.empty_cache()

            print("Finished Prior")

            generation_parameters = {"prompt": p.prompt,
                                    "negative_prompt": p.negative_prompt,
                                    "image_embeds": image_embeds,
                                    "negative_image_embeds": negative_image_embeds,
                                    "height": p.height,
                                    "width": p.width,
                                    "guidance_scale": p.cfg_scale,
                                    "num_inference_steps": p.steps}
            
            all_result_images = []

            pipe = None

            state.job_no = p.n_iter * p.batch_size

            initial_infos = []
            p.all_negative_prompts = [p.negative_prompt] * p.n_iter * p.batch_size
            p.all_prompts = [p.prompt] * p.n_iter * p.batch_size

            for b in range(p.n_iter):
                for batchid in range(p.batch_size):
                    initial_infos.append(self.create_infotext(p,
                                                                p.all_prompts, 
                                                                all_seeds, 
                                                                all_seeds, 
                                                                iteration=b,
                                                                position_in_batch=batchid))

                state.job = "Generating"
                if init_image == None:
                    pipe = getattr(shared, 'pipe', None)

                    if not isinstance(pipe, KandinskyPipeline) or pipe == None:
                        if pipe != None:
                            pipe = None
                            gc.collect()
                            devices.torch_gc()
                        pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", variant="fp16", torch_dtype=torch.float16, cache_dir=self.cache_dir)#, scheduler=dpm)
                        pipe.to("cuda")
                        #pipe.enable_sequential_cpu_offload()
                        pipe.enable_attention_slicing(self.attention_type)
                        pipe.unet.to(memory_format=torch.channels_last)
                        shared.pipe = pipe
                    else:
                        pipe.to("cuda")
                        
                    result_images = pipe(**generation_parameters, num_images_per_prompt=p.batch_size).images


                    if extra_image != [] and extra_image is not None:
                        for i in range(len(result_images)):
                            result_images[i] = self.mix_images(p, pipe, pipe_prior, generation_parameters, img1_strength, img2_strength, inference_steps, prior_cfg_scale, result_images[i], Image.fromarray(extra_image),# self.img2_name,
                                                               torch.Generator().manual_seed(seed + i + b * p.batch_size))

                else:
                    if p.image_mask == None:
                        if not isinstance(pipe, KandinskyImg2ImgPipeline) or pipe == None:
                            if pipe != None:
                                pipe = None
                                gc.collect()
                                devices.torch_gc()

                            pipe = KandinskyImg2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", variant="fp16", torch_dtype=torch.float16, cache_dir=self.cache_dir)#, scheduler=dpm)
                            #pipe.enable_sequential_cpu_offload()
                            pipe.enable_attention_slicing(self.attention_type)
                            pipe.unet.to(memory_format=torch.channels_last)

                        if p.denoising_strength != 0:
                            pipe.to("cuda")
                            result_images = pipe(**generation_parameters, num_images_per_prompt=p.batch_size, image=init_image, strength=p.denoising_strength).images
                        else:
                            result_images = [init_image] * p.batch_size

                        if extra_image != [] and extra_image is not None:
                            pipe.to("cpu")
                            for i in range(len(result_images)):
                                result_images[i] = self.mix_images(p, pipe, pipe_prior, generation_parameters, img1_strength, img2_strength, inference_steps, prior_cfg_scale, result_images[i], Image.fromarray(extra_image),# self.img2_name,
                                                               torch.Generator().manual_seed(seed + i + b * p.batch_size))
                    else:
                        if not isinstance(pipe, KandinskyInpaintPipeline) or pipe == None:
                            if pipe != None:
                                pipe = None
                                gc.collect()
                                devices.torch_gc()
                            pipe = KandinskyInpaintPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", variant="fp16", torch_dtype=torch.float16, cache_dir=self.cache_dir)#, scheduler=dpm)
                            pipe.to("cuda")
                            #pipe.enable_sequential_cpu_offload()
                            pipe.enable_attention_slicing(self.attention_type)
                            pipe.unet.to(memory_format=torch.channels_last)
                        else:
                            pipe.to("cuda")

                        crop_region = None
                        if not p.inpainting_mask_invert:
                            p.image_mask = ImageOps.invert(p.image_mask)

                        if p.mask_blur > 0:
                            p.image_mask = p.image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

                        mask = p.image_mask
                        mask = mask.convert('L')
                        new_init_image = init_image

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

                        result_images = pipe(**generation_parameters, num_images_per_prompt=p.batch_size, image=new_init_image, mask_image=mask).images

                        if extra_image != [] and extra_image is not None:
                            for i in range(len(result_images)):
                                result_images[i] = self.mix_images(p, pipe, pipe_prior, generation_parameters, img1_strength, img2_strength, inference_steps, prior_cfg_scale, result_images[i], Image.fromarray(extra_image),# self.img2_name,
                                                               torch.Generator().manual_seed(seed + i + b * p.batch_size))

                        if p.inpaint_full_res:
                            for i in range(len(result_images)):
                                paste_loc = p.paste_to
                                x, y, w, h = paste_loc
                                base_image = Image.new('RGBA', (init_image.width, init_image.height))
                                mask = ImageOps.invert(mask)
                                result_images[i] = images.resize_image(1, result_images[i], w, h)
                                mask = images.resize_image(1, mask, w, h)
                                mask = mask.convert('L')


                                base_image.paste(result_images[i], (x, y), mask=mask)
                                image = init_image
                                image = image.convert('RGBA')
                                image.alpha_composite(base_image)
                                image.convert('RGB')
                                processing.apply_color_correction(processing.setup_color_correction(init_image), image)
                                result_images[i] = image
                        #else:
                        #    for i in range(len(result_images)):
                        #        base_image = result_images[i]
                        #        base_image = base_image.convert('RGBA')
                        #        mask = ImageOps.invert(mask)
                        #        mask = mask.convert('L')

                        #        image = init_image
                        #        image = image.convert('RGBA')
                        #        image.alpha_composite(base_image)
                        #        image.convert('RGB')
                        #        result_images[i] = image


                for imgid in range(len(result_images)):
                    if type(p.prompt) != list:
                        images.save_image(result_images[imgid], p.outpath_samples, "", all_seeds[imgid], p.prompt[:75], opts.samples_format, info=initial_infos[imgid], p=p)
                    else:
                        images.save_image(result_images[imgid], p.outpath_samples, "", all_seeds[imgid], p.prompt[0][:75], opts.samples_format, info=initial_info[imgid], p=p)

                all_result_images.extend(result_images)

                state.job_no = b * p.batch_size
                state.current_image = result_images[0]
                state.nextjob()

                pipe.to("cpu")

            #del pipe
            del generators
            gc.collect()
            devices.torch_gc()
            torch.cuda.empty_cache()
            initial_info = self.create_infotext(p, p.all_prompts, all_seeds, all_seeds, iteration=0, position_in_batch=0)

            p.n_iter = 1
            state.end()
            #shared.sd_model = None

            #reload_model()
            #sd_models.reload_model_weights()
            #sd_vae.reload_vae_weights()
            #pdb.post_mortem()

            return KProcessed(p, all_result_images, p.seed, initial_info, all_seeds=all_seeds)

        except RuntimeError as e:
            if getattr(shared, 'pipe_prior', None) != None:
                shared.pipe_prior.to("cpu")

            if getattr(shared, 'pipe', None) != None:
                shared.pipe_prior.to("cpu")

            #shared.sd_model = "kandinsky"
            gc.collect()
            #devices.torch_gc()
            torch.cuda.empty_cache()
            if str(e).startswith('CUDA out of memory.'):
                print("OutOfMemoryError: CUDA out of memory.")
            return

    def create_infotext(self, p: StableDiffusionProcessing, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0): # pylint: disable=unused-argument
        index = position_in_batch + iteration * p.batch_size

        generation_params = {
            "Steps": p.steps,
            "Sampler": "DDIM",
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

        self.sd_model_hash = p.extra_generation_params["Model"]

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
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]
