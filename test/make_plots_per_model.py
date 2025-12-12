import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np


def natural_key(key):
    # Helper for natural sorting (e.g., "down_blocks.0.resnets.0")
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
    #print(key_list)
    return key_list

# Define the base folder and file pattern.
model = 'outputs_stable-diffusion-v1-4_controlled'
metrics = 'results'
base_folder = f"outputs_per_model/{model}/averages/{model}"
pattern = os.path.join(base_folder, f"{metrics}_aggregated_fraction_data_with_ci.json")
file_list = glob.glob(pattern, recursive=True)

print(file_list)
# Build a dictionary organized by feature.
# Structure: { feature: { model_name: { 'layers': [...], 'normA': [...], 'normB': [...], 'errA': [...], 'errB': [...] } } }
feature_data = {}

for file_path in file_list:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model_name = os.path.basename(os.path.dirname(file_path))
    text_pairs = data.get("text_pairs", {})
    
    for feature, layers_dict in text_pairs.items():
        if feature not in feature_data:
            feature_data[feature] = {}
        # Get the sorted layer names.
        layer_names = sorted(layers_dict.keys(), key=natural_key)
        layer_names_new = []
        normA_vals, normB_vals = [], []
        errA_vals, errB_vals = [], []
        
        for layer in layer_names:
            len_layer = len(layer.split('_unet'))
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

        feature_data[feature][model_name] = {
            "layers": layer_names_new,
            "normA": normA_vals,
            "normB": normB_vals,
            "errA": errA_vals,
            "errB": errB_vals
        }

# Reorganize the data by model.
models_data = {}
for feature, model_dict in feature_data.items():
    for model_name, d in model_dict.items():
        if model_name not in models_data:
            models_data[model_name] = {}
        models_data[model_name][feature] = d



for model_name, features in models_data.items():
    plt.figure(figsize=(20, 14))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (feature, d) in enumerate(features.items()):
        if feature not in ['S_acc', 'S_pre', 'S_all']:
            continue
        color = color_cycle[i % len(color_cycle)]

        # Original real-space sequence (meanB)
        y = np.array(d["normB"])
        x = np.arange(len(y))

        plt.errorbar(
                x,
                y,
                yerr=np.array(d["errB"]) / 10,
                marker='s',
                linestyle='-',
                linewidth=1,
                capsize=1,
                color=color,
                label=f"{feature}"
        )
        
    # Use naturally ordered layer labels
    first_feature = next(iter(features.values()))
    sorted_layers = sorted(first_feature["layers"], key=natural_key)

    plt.ylim(0,1)
    plt.xticks(range(len(sorted_layers)), sorted_layers, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Normalized Mean Value")

    plt.title(f"Model: {model_name}")
    plt.legend()
    plt.tight_layout()

    out_name = f"{model_name.replace('/', '_')}_values.png"
    plt.savefig(os.path.join(base_folder, out_name))

