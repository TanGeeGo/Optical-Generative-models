from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

class DDPMPipeline_Costum(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet # compile to multigpu case
        if isinstance(unet.config.sample_size, int):
            image_shape = (
                batch_size,
                unet.config.in_channels,
                unet.config.sample_size,
                unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, unet.config.in_channels, *unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        noise_init = image.clone()
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, return_dict=False)[0]
             # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
            
        if output_type == "pil":
            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = image.clamp(-1, 1)

        if not return_dict:
            if output_type == "tensor":
                return (image, noise_init)
            else:
                return (image,)

        return ImagePipelineOutput(images=image)
    
class DDPMPipeline_Costum_ClsEmb(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        class_sample: torch.Tensor = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Sample gaussian noise to begin loop
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet # compile to multigpu case
        if isinstance(unet.config.sample_size, int):
            image_shape = (
                batch_size,
                unet.config.in_channels,
                unet.config.sample_size,
                unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, unet.config.in_channels, *unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        noise_init = image.clone()

        self.scheduler.set_timesteps(num_inference_steps)

        # class embedding
        if class_sample is None:
            class_sample = torch.randint(0, 10, (batch_size,), generator=generator, dtype=torch.int64).to(self.device)
            # class_sample = torch.randint(0, 10, (batch_size,), dtype=torch.int64).to(self.device)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, class_sample, return_dict=False)[0]
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
            
        if output_type == "pil":
            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = image.clamp(-1, 1)

        if not return_dict:
            if output_type == "tensor":
                return (image, noise_init, class_sample)
            else:
                return (image,)

        return ImagePipelineOutput(images=image)