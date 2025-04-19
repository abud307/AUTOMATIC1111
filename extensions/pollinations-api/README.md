# Pollinations.AI API Extension for Stable Diffusion WebUI

This extension integrates [Pollinations.AI](https://pollinations.ai) with the Stable Diffusion WebUI, allowing users to generate images using Pollinations.AI's free, no-signup API instead of local models. This is especially useful for users without high-end GPUs or those who want to try different models without downloading them.

## Features

- Generate images using Pollinations.AI's cloud-based API directly from the WebUI
- No API key or signup required
- Access to multiple image generation models
- Adjustable parameters (width, height, seed, etc.)
- Option to disable the Pollinations logo overlay

## Usage

1. Go to the txt2img or img2img tab
2. Select "Pollinations API" from the Script dropdown
3. Enter your prompt and adjust parameters as needed
4. Click Generate

## Parameters

- **Model**: Select the Pollinations.AI model to use for generation
- **Width**: Width of the generated image (default: 1024)
- **Height**: Height of the generated image (default: 1024)
- **No Logo**: Disable the Pollinations logo overlay
- **Enhance Prompt**: Use an LLM to enhance the prompt with more details

## About Pollinations.AI

Pollinations.AI is a free and open-source generative AI platform that provides:

- Free access to text, image, and audio generation APIs
- No signup or API key required
- Simple URL-based endpoints for easy integration
- OpenAI-compatible interfaces
- HTTPS and CORS support

Visit [Pollinations.AI](https://pollinations.ai) for more information.
