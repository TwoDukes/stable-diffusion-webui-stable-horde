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

from modules import scripts
import json
import os.path

class SettingsManager:
    settings_file = os.path.join(scripts.basedir(), "settings.json")

    def reset_settings(self):
        self.api_endpoint = "https://stablehorde.net/api"
        self.api_key = "0000000000"
        self.censor_nsfw = True
        self.trusted_workers = True
        self.workers = []

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file) as file:
                opts = json.load(file)

            self.api_endpoint = opts["api_endpoint"]
            self.api_key = opts["api_key"]
            self.censor_nsfw = opts["censor_nsfw"]
            self.trusted_workers = opts["trusted_workers"]
            self.workers = opts["workers"]
        else:
            self.reset_settings()

    def save_settings(self):
        opts = {
            "api_endpoint": self.api_endpoint,
            "api_key": self.api_key,
            "censor_nsfw": self.censor_nsfw,
            "trusted_workers": self.trusted_workers,
            "workers": self.workers
        }

        with open(self.settings_file, "w") as file:
            json.dump(opts, file)
