import os
import io
import json
import requests
import urllib.parse
from PIL import Image
from modules import scripts, shared, images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import gradio as gr

class PollinationsAPI(scripts.Script):
    def __init__(self):
        super().__init__()
        self.models = []
        self.fetch_models()

    def fetch_models(self):
        try:
            response = requests.get("https://image.pollinations.ai/models")
            if response.status_code == 200:
                self.models = response.json()
            else:
                print(f"Failed to fetch models: {response.status_code}")
                self.models = ["flux", "sdxl", "pixart", "dalle"]  # Default models
        except Exception as e:
            print(f"Error fetching models: {e}")
            self.models = ["flux", "sdxl", "pixart", "dalle"]  # Default models

    def title(self):
        return "Pollinations API"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Pollinations.AI API", open=False):
                enabled = gr.Checkbox(label="Enable Pollinations.AI API", value=False)
                model = gr.Dropdown(label="Model", choices=self.models, value=self.models[0] if self.models else "flux")
                width = gr.Slider(minimum=256, maximum=2048, step=64, label="Width", value=1024)
                height = gr.Slider(minimum=256, maximum=2048, step=64, label="Height", value=1024)
                no_logo = gr.Checkbox(label="No Logo", value=False)
                enhance = gr.Checkbox(label="Enhance Prompt", value=False)
                private = gr.Checkbox(label="Private (Don't show in public feed)", value=True)
                info = gr.HTML("<p>This extension uses <a href='https://pollinations.ai' target='_blank'>Pollinations.AI</a>'s free, no-signup API to generate images.</p>")

        return [enabled, model, width, height, no_logo, enhance, private]

    def process(
        self,
        p,
        enabled,
        model,
        width,
        height,
        no_logo,
        enhance,
        private,
    ):
        if not enabled:
            return p

        p.do_not_save_grid = True
        p.do_not_save_samples = True

        # Store original parameters
        self.original_prompt = p.prompt
        self.original_negative_prompt = p.negative_prompt
        self.original_seed = p.seed
        self.original_width = p.width
        self.original_height = p.height

        # Override parameters
        p.width = width
        p.height = height

        return p

    def postprocess(self, p, processed, enabled, model, width, height, no_logo, enhance, private):
        if not enabled:
            return processed

        # Restore original parameters for metadata
        p.prompt = self.original_prompt
        p.negative_prompt = self.original_negative_prompt
        p.seed = self.original_seed
        p.width = self.original_width
        p.height = self.original_height

        # Create a new Processed object
        result = Processed(p, [])

        # Generate images using Pollinations API
        for i in range(p.batch_size * p.n_iter):
            try:
                # Prepare parameters
                params = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "seed": p.seed + i if p.seed != -1 else None,
                    "nologo": "true" if no_logo else "false",
                    "enhance": "true" if enhance else "false",
                    "private": "true" if private else "false",
                    "referrer": "stable-diffusion-webui-extension"
                }

                # Prepare prompt
                prompt = p.prompt
                if p.negative_prompt:
                    prompt += " | negative: " + p.negative_prompt

                # Encode prompt
                encoded_prompt = urllib.parse.quote(prompt)

                # Build URL
                url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

                # Make request
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()

                # Convert response to image
                image = Image.open(io.BytesIO(response.content))
                result.images.append(image)

                # Add info to image
                images.save_image(
                    image,
                    p.outpath_samples,
                    "",
                    p.seed + i if p.seed != -1 else -1,
                    p.prompt,
                    opts.samples_format,
                    info=f"Pollinations.AI API, Model: {model}",
                    p=p,
                )

            except Exception as e:
                print(f"Error generating image with Pollinations API: {e}")
                # Add a blank image or error message image
                error_image = Image.new('RGB', (width, height), color=(0, 0, 0))
                result.images.append(error_image)

        return result
