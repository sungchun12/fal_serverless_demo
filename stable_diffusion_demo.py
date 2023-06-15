from fal_serverless import isolated
import io

requirements = [
    "accelerate",
    "diffusers[torch]>=0.10",
    "ftfy",
    "torch",
    "torchvision",
    "transformers",
    "triton",
    "safetensors",
    "xformers==0.0.16",
]


@isolated(requirements=requirements, machine_type="GPU-T4", keep_alive=20)
def generate(prompt: str):
    import torch
    import os
    import base64
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    model_id = "runwayml/stable-diffusion-v1-5"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hugging_face_cache"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        num_inference_steps=20,
        torch_dtype=torch.float16,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    generator = torch.Generator("cuda")
    image = pipe(prompt, generator=generator).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf


image_data = generate("3 epic capybaras standing on a beach")

with open("test.png", "wb") as f:
    f.write(image_data.getvalue())