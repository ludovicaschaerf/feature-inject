import os
import glob
import json
import matplotlib.pyplot as plt

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
    return key_list

# Define the base folder and file pattern.
base_folder = "../outputs_test_large/averages/unknown_model"
pattern = os.path.join(base_folder, "**", "cv_similarities_aggregated_fraction_data_with_ci.json")
file_list = glob.glob(pattern, recursive=True)

# Build a dictionary organized by feature.
# Structure: { feature: { model_name: { 'layers': [...], 'normA': [...], 'normB': [...], 'errA': [...], 'errB': [...] } } }
feature_data = {}
for file_path in file_list:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model_name = data.get("model", os.path.basename(os.path.dirname(file_path)))
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
            total = meanA + meanB
            if total == 0:
                normA, normB, norm_ciA, norm_ciB = 0, 0, 0, 0
            else:
                normA = meanA / total
                normB = meanB / total
                norm_ciA = ciA / total
                norm_ciB = ciB / total
            normA_vals.append(normA)
            normB_vals.append(normB)
            errA_vals.append(norm_ciA)
            errB_vals.append(norm_ciB)
            layer_names_new.append(layer)
        
        feature_data[feature][model_name] = {
            "layers": layer_names_new,
            "normA": normA_vals,
            "normB": normB_vals,
            "errA": errA_vals,
            "errB": errB_vals
        }

# Reorganize the data by model.
# Structure: { model_name: { feature: { ... } } }
models_data = {}
for feature, model_dict in feature_data.items():
    for model_name, d in model_dict.items():
        if model_name not in models_data:
            models_data[model_name] = {}
        models_data[model_name][feature] = d

print(models_data)
# For each model, create a single plot overlaying curves for each feature.
for model_name, features in models_data.items():
    plt.figure(figsize=(20, 14))
    # Use a color cycle to assign a unique color per feature.
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (feature, d) in enumerate(features.items()):
        #if feature != "ssim":
        #    continue  # Only plot "ssim" feature as per request
        color = color_cycle[i % len(color_cycle)]
        x = list(range(len(d["layers"])))
        # Plot normalized meanB for this feature (marker 's').
        plt.errorbar(
            x,
            d["normB"],
            yerr=d["errB"],
            marker='s',
            linestyle='-',
            # line size thicker
            linewidth=1,
            capsize=1,
            color=color,
            label=f"{feature}"
        )
    
    # Use the layer names from the first feature, sorted naturally.
    first_feature = next(iter(features.values()))
    sorted_layers = sorted([x for x in first_feature["layers"]]) #int for dit
    # Reorder the data accordingly
    for feature_data_entry in features.values():
        order = [feature_data_entry["layers"].index(str(l)) for l in sorted_layers]
        for k in ["normA", "normB", "errA", "errB"]:
            feature_data_entry[k] = [feature_data_entry[k][i] for i in order]
        feature_data_entry["layers"] = sorted_layers

    plt.xticks(range(len(sorted_layers)), sorted_layers, rotation=45, ha="right")
    
    plt.title(f"Model: {model_name}")
    plt.xlabel("Layer")
    plt.ylabel("Normalized Mean Value")
    plt.ylim(0, 1)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, f"{model_name.replace('/', '_')}_normalized_values.png"))
