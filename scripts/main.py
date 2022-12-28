# stable-diffusion-webui-stable-horde, a Stable Horde integration to AUTOMATIC1111's Stable Diffusion web UI
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

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

from modules.scripts import Script
from modules.processing import Processed
import modules.shared
import gradio
import requests
import time
import PIL.Image
import base64
from io import BytesIO

class FakeModel:
    sd_model_hash=""

class Main(Script):
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
        stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

        try:
            for k, v in p.override_settings.items():
                setattr(opts, k, v)

            res = self.process_images_inner(p, nsfw, model, seed_variation)
        finally:
            if p.override_settings_restore_afterwards:
                for k, v in stored_opts.items():
                    setattr(opts, k, v)

        return res

    def process_images_inner(self, p, nsfw, model, seed_variation):
        payload = {
            "prompt": "{} ### {}".format(p.prompt, p.negative_prompt) if len(p.negative_prompt) > 0 else p.prompt,
            "params": {
                "sampler_name": self.SAMPLERS.get(p.sampler_name, "k_euler_a"),
                "cfg_scale": p.cfg_scale,
                "denoising_strength": p.denoising_strength if p.denoising_strength is not None else 0,
                "seed": str(p.seed),
                "height": p.height,
                "width": p.width,
                "seed_variation": seed_variation,
                "karras": p.sampler_name in self.KARRAS,
                "steps": p.steps,
                "n": p.n_iter
            }
        }

        if nsfw:
            payload["nsfw"] = True
        elif self.censor_nsfw:
            payload["censor_nsfw"] = True

        if model != "Random":
            payload["models"] = [model]

        if not self.trusted_workers:
            payload["trusted_workers"] = False

        if len(self.workers) > 0:
            payload["workers"] = self.workers

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
            modules.shared.state.job_count = p.n_iter
            modules.shared.state.job_no = 0
            modules.shared.state.sampling_steps = p.steps
            modules.shared.state.sampling_step = 0

            while True:
                status = requests.get("{}/v2/generate/check/{}".format(self.api_endpoint, id))

                if status.status_code == 200:
                    status = status.json()
                    modules.shared.state.job_no = status["finished"]

                    if status["done"]:
                        modules.shared.state.job_no = p.n_iter
                        images = requests.get("{}/v2/generate/status/{}".format(self.api_endpoint, id))

                        if images.status_code == 200:
                            images = images.json()
                            images = images["generations"]
                            images_list = [PIL.Image.open(BytesIO(base64.b64decode(image["img"]))) for image in images]
                            all_seeds = [image["seed"] for image in images]
                            old_model = modules.shared.sd_model
                            modules.shared.sd_model = FakeModel
                            ret = Processed(p, images_list, seed=all_seeds[0], all_seeds=all_seeds)
                            modules.shared.sd_model = old_model
                            return ret
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
