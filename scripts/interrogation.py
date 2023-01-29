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

from modules import script_callbacks
import gradio
import requests
import time
from scripts.settings_manager import SettingsManager

class StableHordeInterrogateError(Exception):
    pass

class Interrogation(SettingsManager):
    def __init__(self):
        self.interrupted = False

    def ui(self):
        with gradio.Blocks(analytics_enabled=False) as interrogation_tab:
            source_image = gradio.Textbox(max_lines=1, label="Source image URL", interactive=True)
            nsfw = gradio.Checkbox(label="NSFW: Returns a true/false boolean depending on whether the image is displaying NSFW imagery or not", interactive=True)
            caption = gradio.Checkbox(label="Caption: Returns a string describing the image", interactive=True)
            interrogation = gradio.Checkbox(label="Interrogation:  Returns a dictionary of key words best describing the image, with an accompanying confidence score", interactive=True)

            with gradio.Row():
                cancel = gradio.Button(value="Cancel")
                interrogate = gradio.Button(value="Interrogate", variant="primary")

            results = gradio.JSON(label="Results")

            def cancel_click():
                self.interrupted = True

            def interrogate_click(source_image, nsfw, caption, interrogation):
                self.interrupted = False

                try:
                    ret = self.run(source_image, nsfw, caption, interrogation)
                except Exception as e:
                    ret = {e.__class__.__name__: str(e)}

                return (gradio.update(value=ret))

            cancel.click(fn=cancel_click)
            interrogate.click(fn=interrogate_click, inputs=[source_image, nsfw, caption, interrogation], outputs=results, scroll_to_output=True)

        return [(interrogation_tab, "Stable Horde Interrogation", "stable_horde_interrogation")]

    def run(self, source_image, nsfw, caption, interrogation):
        if len(source_image) == 0:
            raise StableHordeInterrogateError("Source image URL must be provided")

        payload = {
            "source_image": source_image,
            "forms": []
        }
        self.load_settings()

        if nsfw:
            payload["forms"].append({
                "name": "nsfw"
            })

        if caption:
            payload["forms"].append({
                "name": "caption"
            })

        if interrogation:
            payload["forms"].append({
                "name": "interrogation"
            })

        if len(payload["forms"]) == 0:
            raise StableHordeInterrogateError("At least one option must be chosen")

        if self.interrupted:
            return

        try:
            session = requests.Session()
            id = session.post("{}/v2/interrogate/async".format(self.api_endpoint), headers={"apikey": self.api_key, "Client-Agent": self.CLIENT_AGENT}, json=payload)
            assert id.status_code == 202, "Status Code: {} (expected {})".format(id.status_code, 202)
            id = id.json()
            id = id["id"]
            timeout = 1

            while True:
                if self.interrupted:
                    return self.cancel_interrogate(id)

                try:
                    status = session.get("{}/v2/interrogate/status/{}".format(self.api_endpoint, id), timeout=timeout)
                    assert status.status_code == 200, "Status Code: {} (expected {})".format(status.status_code, 200)
                    status = status.json()

                    if status["state"] == "done":
                        results = {}

                        for form in status["forms"]:
                            results.update(form["result"])

                        return results
                    else:
                        if timeout > 1:
                            timeout //= 2

                        time.sleep(1)
                except requests.Timeout:
                    if timeout >= 60:
                        raise StableHordeInterrogateError("Reached maximum number of retries")

                    timeout *= 2
                    time.sleep(1)
                except AssertionError:
                    status = status.json()
                    raise StableHordeInterrogateError(status["message"])
        except AssertionError:
            id = id.json()
            raise StableHordeInterrogateError(id["message"])

    def cancel_interrogate(self, id):
        status = requests.delete("{}/v2/interrogate/status/{}".format(self.api_endpoint, id), timeout=60)
        status = status.json()
        results = {}

        for form in status["forms"]:
            if form["state"] == "done":
                results.update(form["result"])

        if len(results) > 0:
            return results

script_callbacks.on_ui_tabs(Interrogation().ui)
