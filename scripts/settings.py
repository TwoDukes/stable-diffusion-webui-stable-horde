# Stable Horde for Web UI, a Stable Horde client for AUTOMATIC1111's Stable Diffusion Web UI
# Copyright (C) 2023  Natan Junges <natanajunges@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from modules import script_callbacks, ui
import gradio
from scripts.settings_manager import SettingsManager

class Settings(SettingsManager):
    def ui(self):
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
                self.reset_settings()
                return (gradio.update(value=self.api_endpoint), gradio.update(value=self.api_key), gradio.update(value=self.censor_nsfw), gradio.update(value=self.trusted_workers), gradio.update(value=", ".join(self.workers)))

            def reload_click():
                self.load_settings()
                return (gradio.update(value=self.api_endpoint), gradio.update(value=self.api_key), gradio.update(value=self.censor_nsfw), gradio.update(value=self.trusted_workers), gradio.update(value=", ".join(self.workers)))

            def apply_click(api_endpoint, api_key, censor_nsfw, trusted_workers, workers):
                self.api_endpoint = api_endpoint if len(api_endpoint) > 0 else "https://stablehorde.net/api"
                self.api_key = api_key if len(api_key) > 0 else "0000000000"
                self.censor_nsfw = censor_nsfw
                self.trusted_workers = trusted_workers

                if len(workers) == 0:
                    self.workers = []
                else:
                    self.workers = list(map(lambda w: w.strip(), workers.split(",")))

                self.save_settings()

            show.click(fn=show_click, inputs=show, outputs=[api_key, show])
            reset.click(fn=reset_click, outputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])
            reload.click(fn=reload_click, outputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])
            apply.click(fn=apply_click, inputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])
            settings_tab.load(fn=reload_click, outputs=[api_endpoint, api_key, censor_nsfw, trusted_workers, workers])

        return [(settings_tab, "Stable Horde Settings", "stable_horde_settings")]

script_callbacks.on_ui_tabs(Settings().ui)
