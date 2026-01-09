import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np


def natural_key(key):
    # Helper for natural sorting (e.g., "down_blocks.0.resnets.0")
    if 'unet' in key:
        key = key.split('_unet')[0].replace('unet.','')
        parts = key.split('.')
        key_list = []
        for part in parts:
            try:
                key_list.append(int(part))
            except ValueError:
                key_list.append(part)
        key_list_temp = key_list[-2]
        key_list[-2] = key_list[-1]
        key_list[-1] = key_list_temp
    else:
        key = key.split('_transformer.')[0].replace('transformer.','')
        parts = key.split('.')
        key_list = []
        for part in parts:
            if 'single' in part:
                part = part.replace('single','zingle')
            try:
                key_list.append(int(part))
            except ValueError:
                key_list.append(part)
    return key_list

models = [
    "outputs_kandinsky-2-2-decoder_controlled",
    "outputs_stable-diffusion-2-base_controlled",
    "outputs_stable-diffusion-v1-4_controlled",
    "outputs_stable-diffusion-xl-base-1.0_controlled",
    "outputs_FLUX.1-schnell_controlled",
    "outputs_stable-diffusion-3.5-large-turbo_controlled"
    
]

models_turbo = {
    "outputs_stable-diffusion-2-base_controlled": "outputs_sd-turbo_controlled",
    "outputs_stable-diffusion-xl-base-1.0_controlled": "outputs_sdxl-turbo_controlled",
}

def load_model_into_feature_data(model_id: str, results_dir: str, feature_data: dict, variant_label: str):
    """
    Loads results/<model_id>.json and appends its metrics into feature_data under key:
      feature_data[feature][f"{model_id} ({variant_label})"] = {...}
    """
    file_path = os.path.join(results_dir, f"{model_id}.json")
    if not os.path.exists(file_path):
        print(f"WARNING: missing file: {file_path}")
        return None  # caller can decide what to do

    with open(file_path, "r") as f:
        data = json.load(f)

    text_pairs = data.get("text_pairs", {})

    for feature, layers_dict in text_pairs.items():
        if feature not in feature_data:
            feature_data[feature] = {}

        layer_names = sorted(layers_dict.keys(), key=natural_key)

        layer_names_new = []
        normA_vals, normB_vals = [], []
        errA_vals, errB_vals = [], []

        for layer in layer_names:
            split_key = "_unet" if "_unet" in layer else "_transformer."
            len_layer = len(layer.split(split_key))
            if len_layer > 1:
                continue

            entry = layers_dict[layer]
            meanA = entry["meanA"]
            meanB = entry["meanB"]
            ciA = entry.get("ciA", 0)
            ciB = entry.get("ciB", 0)

            normA_vals.append(meanA)
            normB_vals.append(meanB)
            errA_vals.append(ciA)
            errB_vals.append(ciB)
            layer_names_new.append(layer)

        # store under a display name so we can plot both variants
        display_name = f"{model_id} ({variant_label})"
        feature_data[feature][display_name] = {
            "layers": layer_names_new,
            "normA": normA_vals,
            "normB": normB_vals,
            "errA": errA_vals,
            "errB": errB_vals,
        }

    return file_path


RESULTS_DIR = "results"
OUTPUT_FOLDER = "figures"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for model in models:
    print("Processing", model)

    feature_data = {}

    # ---- load normal ----
    if 'turbo' in model or 'schnell' in model:
        load_model_into_feature_data(model, RESULTS_DIR, feature_data, variant_label="turbo")
    else:
        load_model_into_feature_data(model, RESULTS_DIR, feature_data, variant_label="base")
    
    # ---- load turbo if available ----
    turbo_model = models_turbo.get(model)
    if turbo_model is not None:
        print("  also loading turbo:", turbo_model)
        load_model_into_feature_data(turbo_model, RESULTS_DIR, feature_data, variant_label="turbo")

    # ---- reorganize by "model family" (base model name without variant) ----
    # We want one figure per base `model` in the outer loop, and within it plot base+turbo.
    # So we’ll build `models_data` with just one entry keyed by `model`.
    models_data = {model: {}}

    for feature, model_dict in feature_data.items():
        # model_dict keys are like "outputs_xxx_controlled (base)" and "(turbo)"
        models_data[model][feature] = model_dict

    # ---- plot ----
    for model_family, features in models_data.items():
        plt.figure(figsize=(20, 10))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # ensure consistent color per feature
        plotted_features = [f for f in features.keys() if f in ["color", "ssim",]] #"clip","S_acc", "S_pre"
        plotted_features.sort()

        # add feature S_acc - S_pre
        
        for i, feature in enumerate(plotted_features):
            color = color_cycle[i % len(color_cycle)]

            # features[feature] is a dict: { "model_id (base)": d, "model_id (turbo)": d, ... }
            series_dict = features[feature]

            # plot base first (solid), then turbo (dashed) if present
            # pick by suffix in the display_name
            base_keys = [k for k in series_dict.keys() if k.endswith("(base)")]
            turbo_keys = [k for k in series_dict.keys() if k.endswith("(turbo)")]

            def plot_one(key, linestyle, alpha, label_suffix):
                
                d = series_dict[key]
                y = np.array(d["normB"])
                error = np.array(d["errB"])
                if "S_" not in feature or ('FLUX' in key or '3.5' in key):
                    y = 1 - y
                if "S_" in feature:
                    error = error / 10
                x = np.arange(len(y))

                first_layers = d["layers"]

                plt.errorbar(
                    x,
                    y,
                    yerr=error,
                    marker="s",
                    linestyle=linestyle,
                    linewidth=3,
                    capsize=3,
                    color=color,
                    alpha=alpha,
                    label=f"{feature}{label_suffix}",
                )
                sorted_layers = sorted(first_layers, key=natural_key)
                
                return sorted_layers

            for k in base_keys:
                sorted_layers = plot_one(k, linestyle="-", alpha=1.0, label_suffix=" (base)", )
            for k in turbo_keys:
                sorted_layers = plot_one(k, linestyle="--", alpha=1.0, label_suffix=" (turbo)")

        # if first_layers is None:
        #     print(f"WARNING: nothing plotted for {model_family}")
        #     plt.close()
        #     continue

        
        plt.ylim(0, 1)
        plt.xticks(range(len(sorted_layers)), sorted_layers, rotation=45, ha="right")
        plt.xlabel("Layer")
        plt.ylabel("Normalized Mean Value")
        plt.title(f"Model: {model_family}")
        plt.legend()
        plt.tight_layout()

        out_name = f"{model_family}.png"
        plt.savefig(os.path.join(OUTPUT_FOLDER, out_name))
        plt.close()
