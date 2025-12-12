import os
import sys
import gc
import random
import yaml
import torch
from glob import glob
from argparse import ArgumentParser, Namespace

from main import main   # your existing main() function
sys.path.insert(0, './test')
from prompts import scribblr_prompts, stockimg_prompts, objects, backgrounds, styles


def load_config(path):
    with open(os.path.join("configs", path), "r") as f:
        cfg = yaml.safe_load(f)

    # dtype conversion
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
    }
    cfg["float_"] = dtype_map[cfg.get("float_dtype", "float16")]
    return cfg


def run(config_path):

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps'

    # ---- Load config ----
    config = load_config(config_path)

    model       = config["model"]
    variant     = config["variant"]
    model_name  = config["model_name"]
    image_size  = config["image_size"]
    float_      = config["float_"]

    scale       = config["scale"]
    seed        = config["seed"]
    ddim_steps  = config["ddim_steps"]

    # ---- Prepare test type ----
    test_type = 'controlled'
    output_folder = f'outputs_{model.split("/")[-1]}_{test_type}'
    os.makedirs(output_folder, exist_ok=True)

    num_items = 500 - len(glob(output_folder + '/*'))
    print('running', num_items, 'tests')

    # ---- Build prompt lists ----
    if test_type == 'wild':
        prompts = scribblr_prompts + stockimg_prompts
        sampled_prompts = random.sample(
            [(a, b) for a in prompts for b in prompts if a != b],
            num_items
        )
    else:
        sampled_prompts = []
        for _ in range(num_items):
            obj = random.choice(objects)
            back = random.choice(backgrounds)
            style = random.choice(styles)
            prompt1 = f"A high-resolution image of a {obj} in the {back}, {style}"

            obj2 = random.choice(objects)
            back2 = random.choice(backgrounds)
            style2 = random.choice(styles)
            prompt2 = f"A high-resolution image of a {obj2} in the {back2}, {style2}"

            if (obj != obj2) and (back != back2) and (style != style2) and (prompt1, prompt2) not in sampled_prompts:
                sampled_prompts.append((prompt1, prompt2))

    print(len(sampled_prompts))

    # ---- Main loop ----
    for prompt_A, prompt_B in sampled_prompts:

        sg_tag = f"{'_'.join(prompt_A.split())}_{'_'.join(prompt_B.split())}"
        sg_dir = os.path.join(output_folder, sg_tag)

        if not os.path.isdir(sg_dir):
            os.makedirs(sg_dir, exist_ok=True)
            print(f"\nGenerating A & B for {prompt_A} -> {prompt_B}")

            base_args = {
                'out_dir': output_folder,
                'prompt_A': prompt_A,
                'prompt_B': prompt_B,
                'variant': variant,
                'device': device,
                'image_size': image_size,
                'model': model,
                'model_name': model_name,
                'guidance_scale': 0.0 if ('turbo' in model or 'schnell' in model) else scale,
                'num_inference_steps': ddim_steps,
                'seed': seed,
                'float': float_,
                'timesteps': [1000, 0],
                'switch_guidance': {},
                'selected_skip_keys': '',
            }

            # --- Generate A and B ---
            image_A, image_B, injected_skips, pipe_B = main(
                Namespace(**base_args),
                save_results=False,
                save_b=True
            )

            image_A.save(os.path.join(sg_dir, "A.png"))
            image_B.save(os.path.join(sg_dir, "B.png"))

            # --- Generate C for each individual skip ---
            for layer in injected_skips.keys():
                sample = [layer]
                skip_tag = f"skips_{'_'.join(sample)}"

                hyper_args = base_args.copy()
                hyper_args.update({'selected_skip_keys': sample})

                print(f"Generating C with skip={skip_tag}")
                try:
                    image_C = main(
                        Namespace(**hyper_args),
                        injected_skips=injected_skips,
                        pipe_B=pipe_B,
                        save_results=False
                    )
                    image_C.save(os.path.join(sg_dir, f"C_{skip_tag}.png"))
                except Exception as e:
                    print(e, sample)

            # --- Generate C for random combinations of skips ---
            for _ in range(50):
                n = random.choice([2, 3])
                sample = random.sample(list(injected_skips.keys()), n)
                skip_tag = f"skips_{'_'.join(sample)}"

                hyper_args = base_args.copy()
                hyper_args.update({'selected_skip_keys': sample})

                print(f"Generating C with skip={skip_tag}")
                try:
                    image_C = main(
                        Namespace(**hyper_args),
                        injected_skips=injected_skips,
                        pipe_B=pipe_B,
                        save_results=False
                    )
                    image_C.save(os.path.join(sg_dir, f"C_{skip_tag}.png"))
                except Exception as e:
                    print(e, sample)
                    
            del pipe_B
            gc.collect()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file inside ./configs/"
    )
    args = parser.parse_args()
    run(args.config)