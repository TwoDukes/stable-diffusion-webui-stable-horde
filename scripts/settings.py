# Stable Horde for Web UI, a Stable Horde client for AUTOMATIC1111's Stable Diffusion Web UI
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

from modules import script_callbacks, scripts, ui
import gradio
import json
import os.path

settings_file = os.path.join(scripts.basedir(), "settings.json")

def settings():
    with gradio.Blocks(analytics_enabled=False) as settings_tab:
        api_endpoint = gradio.Textbox(max_lines=1, placeholder="https://stablehorde.net/api", label="API endpoint", interactive=True)

        with gradio.Row():
            api_key = gradio.Textbox(max_lines=1, placeholder="0000000000", label="API key", interactive=True, type="password")
            show = gradio.Button(value="Show", elem_id="horde_show_api_key")

        censor_nsfw = gradio.Checkbox(label="Censor NSFW when NSFW is disabled", interactive=True)
        trusted_workers = gradio.Checkbox(label="Only send requests to trusted workers", interactive=True)
        workers = gradio.Textbox(max_lines=1, label="Only send requests to these workers", interactive=True)

        with gradio.Row():
            reset = gradio.Button(value=ui.reuse_symbol + " Reset settings")
            reload = gradio.Button(value=ui.refresh_symbol + " Reload settings")
            apply = gradio.Button(value=ui.save_style_symbol + " Apply settings", variant="primary")

        def show_click(show):
            return (gradio.update(type="text" if show == "Show" else "password"), gradio.update(value="Hide" if show == "Show" else "Show"))

        def reset_click():
            opts = {
                "api_endpoint": "https://stablehorde.net/api",
                "api_key": "0000000000",
                "censor_nsfw": True,
                "trusted_workers": True,
                "workers": []
            }

            return (gradio.update(value=opts["api_endpoint"]), gradio.update(value=opts["api_key"]), gradio.update(value=opts["censor_nsfw"]), gradio.update(value=opts["trusted_workers"]), gradio.update(value=", ".join(opts["workers"])))

        def reload_click():
            if os.path.exists(settings_file):
                with open(settings_file) as file:
                    opts = json.load(file)

                return (gradio.update(value=opts["api_endpoint"]), gradio.update(value=opts["api_key"]), gradio.update(value=opts["censor_nsfw"]), gradio.update(value=opts["trusted_workers"]), gradio.update(value=", ".join(opts["workers"])))
            else:
                return reset_click()

        def apply_click(api_endpoint, api_key, censor_nsfw, trusted_workers, workers):
            if len(api_endpoint) == 0:
                api_endpoint = "https://stablehorde.net/api"

            if len(api_key) == 0:
                api_key = "0000000000"

            if len(workers) == 0:
                workers = []
            else:
                workers = list(map(lambda w: w.strip(), workers.split(",")))

            opts = {
                "api_endpoint": api_endpoint,
                "api_key": api_key,
                "censor_nsfw": censor_nsfw,
                "trusted_workers": trusted_workers,
                "workers": workers
            }

            with open(settings_file, "w") as file:
                json.dump(opts, file)

        show.click(fn=show_click, inputs=show, outputs=[api_key, show])
        reset.click(fn=reset_click, outputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])
        reload.click(fn=reload_click, outputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])
        apply.click(fn=apply_click, inputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])
        settings_tab.load(fn=reload_click, outputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])

    return [(settings_tab, "Stable Horde Settings", "stable_horde_settings")]

script_callbacks.on_ui_tabs(settings)
