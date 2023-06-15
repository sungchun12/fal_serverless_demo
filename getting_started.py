# Problem I'm solving for: how do I get a basic python function working in the cloud without having to learn a bunch of new stuff?

from fal_serverless import isolated, cached


@cached
def model():
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    return pipe


@isolated(requirements=["diffusers","transformers","torch"], machine_type="GPU-T4")
def predict(prompt):
    pipe = model()
    print("Generating new image")
    return pipe(prompt).images[0]


predict("An astronaut riding a horse")