#!/usr/bin/env python3
"""
experiment.py

Launch a skip injection experiment with configurable parameters via the command line.
The experiment:
  - Runs Pipeline A to capture skip activations from a first prompt.
  - Runs Pipeline C (non-injected) with a second prompt.
  - Runs Pipeline B with skip injection on the second prompt.
All outputs (images and a subplot) and the experiment configuration (saved as a JSON)
are stored in an output folder.
    
Usage:
    python experiment.py --experiment_name my_experiment [other args...]
"""

import argparse
import json
import torch
import gc

from utils.utils_test import (
    setup_pipeline,
    run_pipeline,
    register_skip_hooks,
    remove_skip_hooks,
)


def main(args, injected_skips=None, pipe_B=None, save_results=False, save_b=False):
    
    # Set random seeds for reproducibility.
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.seed()

    # Determine device.
    device = args.device

    # --- Pipeline A: Skip Capture Mode ---
    print("\nInitializing Pipeline A (skip capture mode)...")
    
    if save_b:
        pipe_A = setup_pipeline(args.model_name, args.model, args.device, args.float, variant=args.variant)
        generator = torch.Generator(device).manual_seed(args.seed)
        final_output_A, injected_skips = run_pipeline(
            pipe_A,
            args.prompt_A,
            args.num_inference_steps,
            args.image_size,
            generator,
            args.guidance_scale,
        )
        image_a = final_output_A.images[0]
        
        generator = torch.Generator(device).manual_seed(args.seed)
        print("\nRunning Pipeline C (non-injected mode)...")
        final_output_C, _ = run_pipeline(
                pipe_A,
                args.prompt_B,
                args.num_inference_steps,
                args.image_size,
                generator,
                args.guidance_scale,
        )
        final_image_c = final_output_C.images[0]
       
        del pipe_A
        gc.collect()

        pipe_B = setup_pipeline(args.model_name, args.model, args.device, args.float, variant=args.variant)
             
                  
    if not save_b:
        selected_skip = args.selected_skip_keys
        # Register skip injection hooks on the U-Net modules.
        pipe_B, hook_handles = register_skip_hooks(
                pipe_B,
                injected_skips,
                set(selected_skip),
                args.switch_guidance,
                args.num_inference_steps,
                args.timesteps,
        )
        generator = torch.Generator(device).manual_seed(args.seed)
        final_output_B, _ = run_pipeline(
                pipe_B,
                args.prompt_B,
                args.num_inference_steps,
                args.image_size,
                generator,
                args.guidance_scale,
        )
        final_image_b = final_output_B.images[0]
        remove_skip_hooks(hook_handles)
  
    if save_b:
        return image_a, final_image_c, injected_skips, pipe_B
    else:
        return final_image_b



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skip Injection Experiment Configuration")
    # Experiment identification.
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="skip_injection_experiment",
        help="Name of the experiment",
    )
    # Model and pipeline parameters.
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-schnell", #"PixArt-alpha/PixArt-XL-2-1024-MS", "stabilityai/stable-diffusion-3-medium-diffusers", #stabilityai/stable-diffusion-3.5-large-turbo
        help="Model identifier",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux", #"PixArt-alpha/PixArt-XL-2-1024-MS", "stabilityai/stable-diffusion-3-medium-diffusers", #stabilityai/stable-diffusion-3.5-large-turbo
        help="Model identifier",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help="Model variant",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=3,
        help="Number of diffusion inference steps",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image width and height (in pixels)",
    )
    
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Guidance scale (for classifier-free guidance)",
    )
    # Device and random seed.
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps",
        help="Device to run the experiment (e.g. 'cuda', 'mps', or 'cpu')",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    # Prompts.
    parser.add_argument(
        "--prompt_A",
        type=str,
        default="A high-resolution image of an elephant in the Savannah",
        help="Text prompt for Pipeline A (skip capture)",
    )
    parser.add_argument(
        "--prompt_B",
        type=str,
        default="A high-resolution image of a zebra on the moon",
        help="Text prompt for Pipelines B and C",
    )
    
    parser.add_argument(
        "--prompt_negative",
        type=str,
        default="ugly, blurry, low res, unrealistic",
        help="Negative prompt in the generation",
    )
    
    # Skip injection parameters.
    parser.add_argument(
        "--selected_skip_keys",
        nargs="+",
        default=None,
        help="List of skip keys to inject",
    )
    parser.add_argument(
        "--timesteps",
        nargs=2,
        type=int,
        default=[1000, 0],
        help="Timestep range for injection (two integers)",
    )
    # For switch guidance, pass a JSON-formatted string that maps keys to factors.
    parser.add_argument(
        "--switch_guidance",
        type=json.loads,
        default='{"unet.down_blocks.2.resnets.1": 1, "unet.mid_block.attentions.0": 1}',
        help="Switch guidance factors as a JSON string",
    )

    # Validate parameters
    def validate_parameters(args):
        if args.model == "stabilityai/sdxl-turbo":
            if args.num_inference_steps > 3:
                print("Warning: num_inference_steps should be 3 for model stabilityai/sdxl-turbo.")
            if args.guidance_scale > 0.0:
                print("Warning: guidance_scale should be 0.0 for model stabilityai/sdxl-turbo.")




    args = parser.parse_args()
    validate_parameters(args)
    main(args)
