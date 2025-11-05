#!/usr/bin/env python3
"""
utilities.py

Helper functions for setting up diffusion pipelines, running experiments,
injecting skip activations via hooks, saving images, and plotting results.

Requires:
  - torch
  - numpy
  - matplotlib
  - SDLens (which provides ModifiedStableDiffusionPipeline)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
from argparse import Namespace

from src.hooked_sd_pipeline import ModifiedStableDiffusionPipeline

def setup_pipeline(model_name:str, model: str, device: str, torch_dtype, variant: str = "fp16", use_safetensors: bool = True):
    """
    Initialize and return a diffusion pipeline.
    
    Args:
        model (str): The model identifier (e.g. "stabilityai/sdxl-turbo").
        device (str): The device to load the pipeline on (e.g. "mps", "cuda", or "cpu").
        torch_dtype: The torch data type (e.g. torch.float16).
        variant (str): Variant of the model to use.
        use_safetensors (bool): Whether to use safetensors for model weights.

    Returns:
        An instance of ModifiedStableDiffusionPipeline loaded on the specified device.
    """
    try:
        pipe = ModifiedStableDiffusionPipeline.from_pretrained(
            model_name,
            model,
            torch_dtype=torch_dtype,
            variant=variant,
            use_safetensors=use_safetensors,
        )
    except:
        pipe = ModifiedStableDiffusionPipeline.from_pretrained(
            model_name,
            model,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
        )
    
    
    pipe = pipe.to(device)
    return pipe

def remove_skip_hooks(hook_refs):
    """
    Remove registered forward hooks and clean up attached attributes.

    Args:
        hook_refs (list of (module, handle)): List of modules and their registered hook handles.
    """
    for module, handle in hook_refs:
        handle.remove()
        for attr in ["skip_key", "switch_g", "pippo_counter"]:
            if hasattr(module, attr):
                delattr(module, attr)

def run_pipeline(pipe, prompt: str, num_inference_steps: int, image_size: int, generator, guidance_scale: float):
    """
    Run inference with the given diffusion pipeline.
    
    Args:
        pipe: The diffusion pipeline.
        prompt (str): Text prompt to generate an image.
        num_inference_steps (int): Number of inference steps in the diffusion process.
        image_size (int): The height and width (in pixels) for the generated image.
        generator: Torch random number generator (with a fixed seed for reproducibility).
        guidance_scale (float): Guidance scale (commonly used for classifier-free guidance).

    Returns:
        A tuple (final_output, skip_connections) where final_output typically includes the generated image.
    """
    return pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        height=image_size,
        width=image_size,
        generator=generator,
        guidance_scale=guidance_scale,
    )

def save_image(image, output_filename: str):
    """
    Save the given image (a PIL image) to the specified file.
    
    Args:
        image: The image to save.
        output_filename (str): Path to the output file.
    """
    image.save(output_filename)
    print(f"Saved image as '{output_filename}'.")

def register_skip_hooks(pipe, injected_skips, selected_skip_keys, switch_guidance, num_inference_steps, timesteps):
    """
    Register forward hooks on U-Net modules in the diffusion pipeline to inject skip activations.
    
    The hook will check if the module's key is in the injected_skips dictionary and, if so,
    will replace the module's output with a blend between the original output and the injected
    skip activation (using a guidance factor if specified).

    Args:
        pipe: The diffusion pipeline whose U-Net will be modified.
        injected_skips (dict): Dictionary with skip activation tensors captured from Pipeline A.
        selected_skip_keys (set): Set of keys (module names) at which skip injection is active.
        switch_guidance (dict): Mapping from module keys to a guidance mixing factor.
        num_inference_steps (int): Total number of diffusion inference steps.
        timesteps (list): Two-element list defining the timestep range [start, end] during which injection is applied.

    Returns:
        The diffusion pipeline with hooks registered.
    """
    def skip_injection_hook(module, input, output):
        """
        Hook function applied during the forward pass of selected U-Net modules.
        
        This function:
          - Maintains a per-module counter for the current diffusion step.
          - Checks if the module's key is selected for skip injection.
          - If within the designated timestep range, slices the cached skip activation tensor
            to the current step and injects it into the output (optionally blended with guidance).
        """
        # Initialize or increment the per-module diffusion step counter.
        if not hasattr(module, "pippo_counter"):
            module.pippo_counter = 0
        else:
            module.pippo_counter += 1
        step_index = module.pippo_counter

        key = getattr(module, "skip_key", None)
        if key is not None and key in injected_skips:
            # Only proceed if this module's key is in the selected set.
            if selected_skip_keys is not None and key not in selected_skip_keys:
                return output
            
            # Calculate the lower and upper bounds for injection based on timesteps.
            lower_bound = np.floor((1000 - timesteps[0]) / 1000 * num_inference_steps)
            upper_bound = np.ceil((1000 - timesteps[1]) / 1000 * num_inference_steps)
            
            if step_index < lower_bound or step_index > upper_bound:
                print(f"Step index {step_index} out of injection range for key '{key}'.")
                return output

            # Retrieve the cached skip activation tensor for this module.
            injected = injected_skips[key]
            if injected.ndim >= 2 and injected.shape[1] > step_index:
                # Extract the slice corresponding to the current diffusion step.
                injected_slice = injected[:, step_index:step_index + 1, ...]
            else:
                print("Warning: injected tensor does not have enough steps; using full tensor.")
                injected_slice = injected

            # If a guidance mixing factor is specified, blend the outputs.
            if hasattr(module, "switch_g") and module.switch_g is not None:
                # Assume output is a tuple and the first element holds the relevant tensor.
                injected_slice = output[0] + module.switch_g * (injected_slice - output[0])

            # Adjust the output based on its type (tuple or tensor).
            if isinstance(output, tuple):
                expected_shape = output[0].shape
                if injected_slice.ndim == len(expected_shape) + 1:
                    injected_slice = injected_slice.squeeze(1)
                if injected_slice.shape != expected_shape:
                    print(f"Warning: injected slice shape {injected_slice.shape} != expected {expected_shape}.")
                return (injected_slice,) + output[1:]
            else:
                expected_shape = output.shape
                if injected_slice.ndim == len(expected_shape) + 1:
                    injected_slice = injected_slice.squeeze(1)
                if injected_slice.shape != expected_shape:
                    print(f"Warning: injected slice shape {injected_slice.shape} != expected {expected_shape}.")
                return injected_slice

        # If this module is not selected for injection, return the original output.
        return output
        
    hook_handles = []
    if hasattr(pipe, 'unet'):
        for name, module in pipe.unet.named_modules():
            key = f"unet.{name}"
            if key in injected_skips.keys():
                module.skip_key = key  # Save the module key for use in the hook.
                module.switch_g = (
                    switch_guidance if isinstance(switch_guidance, float)
                    else switch_guidance.get(key, None)
                )
                handle = module.register_forward_hook(skip_injection_hook)
                hook_handles.append((module, handle))
    
    if hasattr(pipe, 'transformer'):
        for name, module in pipe.transformer.named_modules():
            key = f"transformer.{name}"
            if key in injected_skips.keys():
                module.skip_key = key  # Save the module key for use in the hook.
                module.switch_g = (
                    switch_guidance if isinstance(switch_guidance, float)
                    else switch_guidance.get(key, None)
                )
                handle = module.register_forward_hook(skip_injection_hook)
                hook_handles.append((module, handle))

    return pipe, hook_handles

def plot_results(image_a, final_image, final_image_B, output_filename: str):
    """
    Plot three images side-by-side and save the resulting figure.
    
    The subplot displays:
      - The image from Pipeline A (skip capture).
      - The image from Pipeline B (skip injection).
      - The image from Pipeline C (non-injected run).

    Args:
        image_a: PIL image from Pipeline A.
        final_image: PIL image from Pipeline B (with injection).
        final_image_B: PIL image from Pipeline C (without injection).
        output_filename (str): File path where the subplot will be saved.
    """
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(image_a)
    axes[1].imshow(final_image)
    axes[2].imshow(final_image_B)
    fig.suptitle(os.path.basename(output_filename).replace(".png", ""))
    plt.savefig(output_filename)
    print(f"Saved subplot image as '{output_filename}'.")
    plt.close()

def generate_triplet(
    source_prompt: str,
    target_prompt: str,
    output_folder: str,
    model: str,
    model_name: str,
    variant: str,
    device: str,
    image_size: int,
    selected_skip_keys: list,
    main_fn,
    float_: bool = False,
    scale: float = 7.5,
    seed: int = 0,
    ddim_steps: int = 50,
):
    """
    Generate a triplet of images (original, edit, and target) for two prompts.
    
    Args:
        source_prompt (str): The source/original prompt.
        target_prompt (str): The target/edit prompt.
        output_folder (str): Directory to save results.
        model (str): Model identifier (e.g., 'sd15', 'turbo', etc.).
        model_name (str): Model name used by the diffusion pipeline.
        variant (str): Model variant.
        device (str): Device ('cuda', 'cpu', etc.).
        image_size (int): Output image size.
        selected_skip_keys (list): List of skip keys for the model.
        main_fn (callable): The main generation function.
        float_ (bool): Whether to use float precision.
        scale (float): Guidance scale.
        seed (int): Random seed.
        ddim_steps (int): Number of diffusion steps.
    """

    
    # Adjust parameters based on model type
    if 'turbo' in model or 'schnell' in model:
        scale = 0.0
        ddim_steps = 3

    if 'kandinsky' in model:
        ddim_steps = min(ddim_steps, 30)

    print(f"\n--- Generating Triplet ---")
    print(f"scale={scale}, seed={seed}, ddim_steps={ddim_steps}")
    print(f"source_prompt='{source_prompt}'")
    print(f"target_prompt='{target_prompt}'")
   
    # ---- Generate A & B ----
    base_args = {
        'out_dir': output_folder,
        'prompt_A': target_prompt,
        'variant': variant,
        'device': device,
        'prompt_B': source_prompt,
        'image_size': image_size,
        'model': model,
        'model_name': model_name,
        'guidance_scale': scale,
        'num_inference_steps': ddim_steps,
        'seed': seed,
        'float': float_,
        'timesteps': [1000, 0],
        'switch_guidance': {},
        'selected_skip_keys': selected_skip_keys[0]
    }

    print(f"Generating A & B for '{source_prompt}' -> '{target_prompt}'")
    image_A, image_B, injected_skips, pipe_B = main_fn(Namespace(**base_args), save_results=False, save_b=True)
    
    # ---- Generate C ----
    switch_guidance_list = [{}]
    timestep_list = [[1000, 200]]

    for skips in selected_skip_keys:
        skip_tag = f"skips_{'_'.join([s.split('.')[-1] +'_' + s.split('.')[-3] for s in skips])}"

        for sg, ts in zip(switch_guidance_list, timestep_list):
            hyper_args = base_args.copy()
            hyper_args.update({
                'switch_guidance': sg,
                'timesteps': ts,
                'selected_skip_keys': skips
            })

            print(f"Generating C with skip={skip_tag}, SG={sg}, timesteps={ts}")
            image_C = main_fn(Namespace(**hyper_args), injected_skips=injected_skips, pipe_B=pipe_B, save_results=False)
            os.makedirs(output_folder, exist_ok=True)
            plot_results(
                image_A,
                image_B,
                image_C,
                os.path.join(
                    output_folder,
                    f"triplet_seed{seed}_steps{ddim_steps}_scale{scale}_{skip_tag}.png"
                )
            )
            
            
    del pipe_B
    gc.collect()

    