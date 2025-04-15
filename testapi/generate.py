import requests
import base64

url = "http://127.0.0.1:7860"

payload = {
  "prompt": "",
  "negative_prompt": "",
  "styles": [
    "string"
  ],
  "seed": -1,
  "subseed": -1,
  "subseed_strength": 0,
  "seed_resize_from_h": -1,
  "seed_resize_from_w": -1,
  "sampler_name": "string",
  "scheduler": "string",
  "batch_size": 1,
  "n_iter": 1,
  "steps": 50,
  "cfg_scale": 7,
  "width": 512,
  "height": 512,
  "restore_faces": True,
  "tiling": True,
  "do_not_save_samples": False,
  "do_not_save_grid": False,
  "eta": 0,
  "denoising_strength": 0,
  "s_min_uncond": 0,
  "s_churn": 0,
  "s_tmax": 0,
  "s_tmin": 0,
  "s_noise": 0,
  "override_settings": {},
  "override_settings_restore_afterwards": True,
  "refiner_checkpoint": "string",
  "refiner_switch_at": 0,
  "disable_extra_networks": False,
  "firstpass_image": "string",
  "comments": {},
  "enable_hr": False,
  "firstphase_width": 0,
  "firstphase_height": 0,
  "hr_scale": 2,
  "hr_upscaler": "string",
  "hr_second_pass_steps": 0,
  "hr_resize_x": 0,
  "hr_resize_y": 0,
  "hr_checkpoint_name": "string",
  "hr_sampler_name": "string",
  "hr_scheduler": "string",
  "hr_prompt": "",
  "hr_negative_prompt": "",
  "force_task_id": "string",
  "sampler_index": "Euler",
  "script_name": "string",
  "script_args": [],
  "send_images": True,
  "save_images": False,
  "alwayson_scripts": {},
  "infotext": "string"
}

response = requests.post(f"{url}/sdapi/v1/txt2img", json=payload)

# Parse the response
r = response.json()

# Decode and save the image
with open("testapi/output1.png", "wb") as f:
    f.write(base64.b64decode(r['images'][0]))

print("✅ Image saved as output.png")
