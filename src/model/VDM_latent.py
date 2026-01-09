# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import List, Optional, Union

import numpy as np
import torch

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, DDPMScheduler
from data.normalization import get_normalized_body, flatten_embeddings
from data.structures import GarmentDict, Mesh
from body_models.smpl_model import BodyModel
from losses.losses import collision_loss
from utils.geometry import batch_compute_points_normals, l2dist
from utils.uv_tools import apply_displacement
from utils.helpers import batchify_dict

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class VDMStableDiffusionPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
):
    r"""
    Pipeline for D-Garment model inspired by Stable Diffusion pipeline.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers
    ):
        super().__init__()

        unet.requires_grad_(False).eval()
        vae.decoder.requires_grad_(False).eval()
        vae.encoder.requires_grad_(False).eval()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.permute(0, 2, 3, 1)
        return image.float()

    def encode_latents(self, image):
        image = image.permute(0,3,1,2)
        latents = self.vae.encode(image, return_dict=False)[0].mode()
        latents = self.vae.config.scaling_factor * latents
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, generator, latents=None):

        device = self._execution_device

        if latents is None:
            shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            latents = randn_tensor(shape, generator=generator, device=device, dtype=self.unet.dtype)
        else:
            latents = latents.to(device, dtype=self.unet.dtype)
        return latents

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def compile(self):
        # compile the models to potentially improve performance, note that it uses experimental features and that some operations could not be suported by some GPU. Use at your own risks

        # free lunch tuning
        self.fuse_qkv_projections()
        self.unet.to(memory_format=torch.channels_last)
        self.vae.decoder.to(memory_format=torch.channels_last)
        self.vae.encoder.to(memory_format=torch.channels_last)
        # self.enable_vae_slicing() # uncomment in case the GPU is memory limited

        # obscure inductor configuration parameters
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        # full speed ahead! takes long time to compile and is not very robust, waiting for future pytorch updates
        # self.unet.compile(mode="max-autotune", fullgraph=True, dynamic=True)
        # self.vae.decoder.compile(mode="max-autotune", fullgraph=True, dynamic=True)

        # safer compilation but with less performance gain
        self.unet.compile(fullgraph=True, dynamic=False, mode='reduce-overhead')
        self.vae.decoder.compile(fullgraph=True, dynamic=False)

    @torch.enable_grad()
    def guidance_func(self, latents, noise_pred_orig, t, acceleration_weight, clothing_weight, template, body_vertices, body_normal, target=None):
        latents = latents.requires_grad_()

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DDPMScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred_orig) / alpha_prod_t ** 0.5
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")
        
        image = self.decode_latents(pred_original_sample)

        vertices = apply_displacement(template, image)

        clothing = collision_loss(vertices, body_vertices, body_normal)
        loss = clothing.sum() * clothing_weight
    
        if target is not None and acceleration_weight > 0:
            dist = l2dist(vertices, target)
            accel_loss = dist.mean()
            loss += accel_loss * acceleration_weight

        grad = torch.autograd.grad(loss, latents)[0]
        new_latent = noise_pred_orig - torch.sqrt(beta_prod_t) * grad

        return new_latent.detach()

    def __call__(
        self,
        condition: GarmentDict = None,
        template: Mesh = None,
        body_model: BodyModel = None,
        use_material: bool = True,
        acceleration_weight: float = 0,
        clothing_weight: float = 0,
        target: torch.Tensor = None,
        opti_warmup=1,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        decoded: Optional[bool] = True,
        gradient=False,
        gradient_unet=False,
        normalize_material=True
    ):
        r"""
        The call function to the pipeline for generation.
        Args:
            condition (`GarmentDict`, *optional*):
                The informations to guide image generation.
            template (`Mesh`, *optional*):
                The template mesh, needed for guidance.
            body_model (`BodyModel`, *optional*):
                The SMPL body model, needed for guidance.
            use_material (`bool`, *optional*):
                Wether to include cloth material in condition or not.
            acceleration_weight (`float`, *optional*):
                Guidance weight to regularize acceleration (actually velocity).
            clothing_weight (`float`, *optional*):
                Guidance weight to regularize cloth-body penetration.
            target (`torch.Tensor`, *optional*):
                Last vertices positions frame to compute the acceleration (actually velocity).
            opti_warmup (`int`, *optional*, defaults to 1):
                The number of denoising steps without guidance before applying it.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts or optimization.
                If not provided, a latents tensor is generated by sampling using the supplied random `generator`.
            decoded (`bool`, *optional*, defaults to `"True"`):
                Choose between `latent` or decoded output.
            gradient (`bool`, *optional*, defaults to `"False"`):
                Keep gradient for optimization process or not for faster run time. Disables gradients for the whole pipeline including U-net and VAE decoder.
            gradient_unet (`bool`, *optional*, defaults to `"False"`):
                Keep gradient of the U-net for optimization process or not for faster run time.
                Setting `gradient` to True and this to False allows to backpropagate through the scheduler only
                bypassing the U-net. Set to True to optimize cloth material parameters.
        Returns:
            torch.Tensor: 
                Denoised and decoded image prediction if `decoded` is True else denoised latents
        """

        device = self._execution_device

        prev_gradient_context = torch.is_grad_enabled()
        torch.set_grad_enabled(gradient)
        self._interrupt = False
        use_guidance = body_model and template and condition and (acceleration_weight!=0 or clothing_weight!=0)

        # 1. Prepare data
        if condition is not None:
            if len(condition["trans"].shape) == 2:
                condition = batchify_dict(condition)

            if use_guidance:
                bodict = {}
                bodict["betas"] = condition["betas"]
                bodict["trans"], bodict["poses"] = get_normalized_body(condition)
                bodict["trans"] = bodict["trans"][:, -1]
                bodict["poses"] = bodict["poses"][:, -1]
                body_vertices = body_model(bodict)
                body_faces = body_model.get_faces()
                del bodict
                body_normal = batch_compute_points_normals(body_vertices, body_faces)

            condition_embedding = flatten_embeddings(condition, material=use_material, do_normalize=normalize_material).to(device, self.unet.dtype)
            batch_size = condition_embedding.shape[0]
        else:
            condition_embedding = None
            batch_size = 1

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size,
            generator,
            latents,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latents = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                if gradient_unet:
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=condition_embedding,
                        return_dict=False,
                    )[0]
                else:
                    with torch.no_grad(): # block condition gradients but is faster and seems to converge better
                        noise_pred = self.unet(
                            latents,
                            t,
                            encoder_hidden_states=condition_embedding,
                            return_dict=False,
                        )[0]

                if use_guidance and i > opti_warmup:
                    noise_pred = self.guidance_func(latents, noise_pred, t, acceleration_weight, clothing_weight, template, body_vertices, body_normal, target)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if decoded:
            image = self.decode_latents(latents)
        else:
            image = latents
        
        # Offload all models
        self.maybe_free_model_hooks()
        torch.set_grad_enabled(prev_gradient_context)

        return image
