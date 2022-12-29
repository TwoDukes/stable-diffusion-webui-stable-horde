# stable-diffusion-webui-stable-horde, a Stable Horde client integration to AUTOMATIC1111's Stable Diffusion web UI
# Copyright (C) 2022  Natan Junges <natanajunges@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from modules import scripts, processing, shared, images, devices
import gradio
import requests
import time
import PIL.Image
import base64
import io
import os.path
import numpy
import itertools
import torch

class FakeModel:
    sd_model_hash=""

class Main(scripts.Script):
    TITLE = "Run with Stable Horde"
    SAMPLERS = {
        "LMS": "k_lms",
        "LMS Karras": "k_lms",
        "Heun": "k_heun",
        "Euler": "k_euler",
        "Euler a": "k_euler_a",
        "DPM2": "k_dpm_2",
        "DPM2 Karras": "k_dpm_2",
        "DPM2 a": "k_dpm_2_a",
        "DPM2 a Karras": "k_dpm_2_a",
        "DPM fast": "k_dpm_fast",
        "DPM adaptive": "k_dpm_adaptive",
        "DPM++ 2S a": "k_dpmpp_2s_a",
        "DPM++ 2S a Karras": "k_dpmpp_2s_a",
        "DPM++ 2M": "k_dpmpp_2m",
        "DPM++ 2M Karras": "k_dpmpp_2m"
    }
    KARRAS = {"LMS Karras", "DPM2 Karras", "DPM2 a Karras", "DPM++ 2S a Karras", "DPM++ 2M Karras"}
    #settings tab
    api_endpoint = "https://stablehorde.net/api"
    api_key = "0000000000"
    censor_nsfw = True
    trusted_workers = True
    workers = []

    def title(self):
        return self.TITLE

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        nsfw = gradio.Checkbox(False, label="NSFW")
        models = requests.get("{}/v2/status/models".format(self.api_endpoint))

        if models.status_code == 200:
            models = models.json()
            models.sort(key=lambda m: (-m["count"], m["name"]))
            models = ["{} ({})".format(m["name"], m["count"]) for m in models]
        else:
            models = []

        models.insert(0, "Random")
        model = gradio.Dropdown(models, value="Random", label="Model")
        seed_variation = gradio.Number(value=1, label="Seed variation", precision=0)
        return [nsfw, model, seed_variation]

    def run(self, p, nsfw, model, seed_variation):
        if model != "Random":
            model = model.split("(")[0].rstrip()

        return self.process_images(p, nsfw, model, seed_variation)

    def process_images(self, p, nsfw, model, seed_variation):
        # Copyright (C) 2022  AUTOMATIC1111

        stored_opts = {k: shared.opts.data[k] for k in p.override_settings.keys()}

        try:
            for k, v in p.override_settings.items():
                setattr(shared.opts, k, v)

            res = self.process_images_inner(p, nsfw, model, seed_variation)
        finally:
            if p.override_settings_restore_afterwards:
                for k, v in stored_opts.items():
                    setattr(shared.opts, k, v)

        return res

    def process_images_inner(self, p, nsfw, model, seed_variation):
        # Copyright (C) 2022  AUTOMATIC1111

        if type(p.prompt) == list:
            assert(len(p.prompt) > 0)
        else:
            assert p.prompt is not None

        devices.torch_gc()
        seed = processing.get_fixed_seed(p.seed)
        p.subseed = -1
        p.subseed_strength = 0
        p.seed_resize_from_h = 0
        p.seed_resize_from_w = 0

        if type(p.prompt) == list:
            p.all_prompts = list(itertools.chain.from_iterable((p.batch_size * [shared.prompt_styles.apply_styles_to_prompt(p.prompt[x * p.batch_size], p.styles)] for x in range(p.n_iter))))
        else:
            p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

        if type(p.negative_prompt) == list:
            p.all_negative_prompts = list(itertools.chain.from_iterable((p.batch_size * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt[x * p.batch_size], p.styles)] for x in range(p.n_iter))))
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

        if type(seed) == list:
            p.all_seeds = list(itertools.chain.from_iterable(([seed[x * p.batch_size] + y * seed_variation for y in range(p.batch_size)] for x in range(p.n_iter))))
        else:
            p.all_seeds = [int(seed) + x * seed_variation for x in range(len(p.all_prompts))]

        p.all_subseeds = [-1 for _ in range(len(p.all_prompts))]

        def infotext(iteration=0, position_in_batch=0):
            old_model = shared.sd_model
            shared.sd_model = FakeModel
            ret = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, {}, iteration, position_in_batch)
            shared.sd_model = old_model
            return ret

        with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
            old_model = shared.sd_model
            shared.sd_model = FakeModel
            processed = processing.Processed(p, [], p.seed, "")
            file.write(processed.infotext(p, 0))
            shared.sd_model = old_model

        if p.scripts is not None:
            p.scripts.process(p)

        infotexts = []
        output_images = []

        with torch.no_grad():
            if shared.state.job_count == -1:
                shared.state.job_count = p.n_iter

            for n in range(p.n_iter):
                if shared.state.skipped:
                    shared.state.skipped = False

                if shared.state.interrupted:
                    break

                prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
                subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

                if len(prompts) == 0:
                    break

                if p.scripts is not None:
                    p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

                if p.n_iter > 1:
                    shared.state.job = f"Batch {n+1} out of {p.n_iter}"

                x_samples_ddim = self.process_batch_horde(p, nsfw, model, seed_variation, prompts[0], negative_prompts[0], seeds[0])

                if x_samples_ddim is None:
                    del x_samples_ddim
                    devices.torch_gc()
                    break

                if p.scripts is not None:
                    p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * numpy.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(numpy.uint8)
                    image = PIL.Image.fromarray(x_sample)

                    if p.color_corrections is not None and i < len(p.color_corrections):
                        if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_color_correction:
                            image_without_cc = processing.apply_overlay(image, p.paste_to, i, p.overlay_images)
                            images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")

                        image = processing.apply_color_correction(p.color_corrections[i], image)

                    image = processing.apply_overlay(image, p.paste_to, i, p.overlay_images)

                    if shared.opts.samples_save and not p.do_not_save_samples:
                        images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=infotext(n, i), p=p)

                    text = infotext(n, i)
                    infotexts.append(text)

                    if shared.opts.enable_pnginfo:
                        image.info["parameters"] = text

                    output_images.append(image)

                del x_samples_ddim
                devices.torch_gc()
                shared.state.job_no += 1
                shared.state.sampling_step = 0
                shared.state.current_image_sampling_step = 0

            p.color_corrections = None
            index_of_first_image = 0
            unwanted_grid_because_of_img_count = len(output_images) < 2 and shared.opts.grid_only_if_multiple

            if (shared.opts.return_grid or shared.opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
                grid = images.image_grid(output_images, p.batch_size)

                if shared.opts.return_grid:
                    text = infotext()
                    infotexts.insert(0, text)

                    if shared.opts.enable_pnginfo:
                        grid.info["parameters"] = text

                    output_images.insert(0, grid)
                    index_of_first_image = 1

                if shared.opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=infotext(), short_filename=not shared.opts.grid_extended_filename, p=p, grid=True)

        devices.torch_gc()
        old_model = shared.sd_model
        shared.sd_model = FakeModel
        res = processing.Processed(p, output_images, p.all_seeds[0], infotext(), subseed=-1, index_of_first_image=index_of_first_image, infotexts=infotexts)
        shared.sd_model = old_model

        if p.scripts is not None:
            p.scripts.postprocess(p, res)

        return res

    def process_batch_horde(self, p, nsfw, model, seed_variation, prompt, negative_prompt, seed):
        payload = {
            "prompt": "{} ### {}".format(prompt, negative_prompt) if len(negative_prompt) > 0 else prompt,
            "params": {
                "sampler_name": self.SAMPLERS.get(p.sampler_name, "k_euler_a"),
                "cfg_scale": p.cfg_scale,
                "denoising_strength": p.denoising_strength if p.denoising_strength is not None else 0,
                "seed": str(seed),
                "height": p.height,
                "width": p.width,
                "seed_variation": seed_variation,
                "karras": p.sampler_name in self.KARRAS,
                "steps": p.steps,
                "n": p.batch_size
            }
        }

        if nsfw:
            payload["nsfw"] = True
        elif self.censor_nsfw:
            payload["censor_nsfw"] = True

        if not self.trusted_workers:
            payload["trusted_workers"] = False

        if len(self.workers) > 0:
            payload["workers"] = self.workers

        if model != "Random":
            payload["models"] = [model]

        #img2img/inpainting

        post_processing = []

        #upscale RealESRGAN_x4plus

        if p.restore_faces:
            #CodeFormers
            post_processing.append("GFPGAN")

        if len(post_processing) > 0:
            payload["params"]["post_processing"] = post_processing

        id = requests.post("{}/v2/generate/async".format(self.api_endpoint), headers={"apikey": self.api_key}, json=payload)

        if id.status_code == 202:
            id = id.json()
            id = id["id"]
            shared.state.sampling_steps = p.batch_size

            while True:
                status = requests.get("{}/v2/generate/check/{}".format(self.api_endpoint, id))

                if status.status_code == 200:
                    status = status.json()
                    shared.state.sampling_step = status["finished"]

                    if status["done"]:
                        shared.state.sampling_step = shared.state.sampling_steps
                        images = requests.get("{}/v2/generate/status/{}".format(self.api_endpoint, id))

                        if images.status_code == 200:
                            images = images.json()
                            images = images["generations"]
                            images = [PIL.Image.open(io.BytesIO(base64.b64decode(image["img"]))) for image in images]
                            images = [numpy.moveaxis(numpy.array(image).astype(numpy.float32) / 255.0, 2, 0) for image in images]
                            images = [torch.from_numpy(image) for image in images]
                            images = torch.stack(images).to(shared.device)
                            return images
                        else:
                            images = images.json()
                            print(images["message"])
                            break
                    elif status["faulted"]:
                        print("faulted")
                        break
                    elif not status["is_possible"]:
                        print("not is_possible")
                        break
                    else:
                        time.sleep(1)
                else:
                    status = status.json()
                    print(status["message"])
                    break
        else:
            id = id.json()
            print(id["message"])
            print(payload)
