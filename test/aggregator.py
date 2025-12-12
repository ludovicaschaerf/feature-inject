import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.preprocessing import RobustScaler

def confidence_interval(data, confidence=0.95):
    """
    Computes the mean and confidence interval width for 'data' using Student's t-distribution.
    Returns (mean, ci_half_width).
    """
    a = np.array(data, dtype=float)
    n = len(a)
    if n < 2:
        return float(np.mean(a)), 0.0  # If not enough data, return mean with 0 error
    mean = np.mean(a)
    sem = scipy.stats.sem(a)  # standard error
    t_val = scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    ci_half_width = sem * t_val
    return float(mean), float(ci_half_width)

def main(which_sim):
    outputs_dir = "../outputs_per_model/outputs_stable-diffusion-xl-base-1.0_controlled/"

    model_data = {}

    missing = 0
    # 1) Collect data from each subfolder's similarities.json
    for subfolder in os.listdir(outputs_dir):
        subfolder_path = os.path.join(outputs_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            print('Skipping as not a path')
            continue

        similarities_path = os.path.join(subfolder_path, which_sim)
        if not os.path.isfile(similarities_path):
            missing += 1
            continue

        with open(similarities_path, "r") as f:
            similarities = json.load(f)

        model_name = similarities.get("model", "unknown_model")
        if model_name not in model_data:
            model_data[model_name] = {
                "refs": {},
                "text_pairs": {},
                "count": 0
            }

        # 2) Accumulate reference image similarities
        text_sims = similarities.get(which_sim.split('.')[0], {})
        for layer_name, pair_dict in text_sims.items():
            if 'new' in which_sim:
                for pair_key in ['lpips', 'gram', 'ms_ssim']:
                    model_data[model_name]["text_pairs"].setdefault(pair_key, {})
                    simA = pair_dict[pair_key + '_A']
                    simB = pair_dict[pair_key + '_B']
                    
                    model_data[model_name]["text_pairs"][pair_key].setdefault(layer_name, []).append((simA, simB))
                    model_data[model_name]["refs"].setdefault(layer_name, []).append((simA, simB))
            else:
                for pair_key, pair_vals in pair_dict.items():
                    simA = pair_vals.get("simA", 0.0)
                    simB = pair_vals.get("simB", 0.0)
                    model_data[model_name]["text_pairs"].setdefault(pair_key, {})
                    model_data[model_name]["text_pairs"][pair_key].setdefault(layer_name, []).append((simA, simB))
                    model_data[model_name]["refs"].setdefault(layer_name, []).append((simA, simB))
        model_data[model_name]["count"] += 1
            
        # 2) Accumulate reference image similarities
        image_sims = similarities.get("image_similarities", {})
        for layer_name, sim_dict in image_sims.items():
            simA = sim_dict.get("simA", 0.0)
            simB = sim_dict.get("simB", 0.0)
            model_data[model_name]["refs"].setdefault(layer_name, []).append((simA, simB))
        model_data[model_name]["count"] += 1

        # 3) Accumulate text similarities
        text_sims = similarities.get("text_similarities", {})
        for layer_name, pair_dict in text_sims.items():
            for pair_key, pair_vals in pair_dict.items():
                simA = pair_vals.get("simA", 0.0)
                simB = pair_vals.get("simB", 0.0)
                model_data[model_name]["text_pairs"].setdefault(pair_key, {})
                model_data[model_name]["text_pairs"][pair_key].setdefault(layer_name, []).append((simA, simB))

    print(missing)
    # 4) For each model, compute fraction means & confidence intervals, produce stacked bar plots
    for model_name, data in model_data.items():
        model_out_dir = os.path.join(outputs_dir, "averages", model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        # -- A) References --
        # data["refs"] is { layer_name: [(simA, simB), (simA, simB), ...], ... }
        layer_names_refs = sorted(data["refs"].keys())
        fractionA_means_refs = []
        fractionA_cis_refs = []
        fractionB_means_refs = []
        fractionB_cis_refs = []

        aggregated_fractions_refs = {}

        for layer_name in layer_names_refs:
            sim_list = data["refs"][layer_name]  # list of (simA, simB)
            
            # Extract values
            simA_vals = [a for (a, b) in sim_list]
            simB_vals = [b for (a, b) in sim_list]
        
            # Now compute mean + CI on the standardized values
            meanA, ciA = confidence_interval(simA_vals, confidence=0.90)
            meanB, ciB = confidence_interval(simB_vals, confidence=0.90)
            
            fractionA_means_refs.append(meanA)
            fractionA_cis_refs.append(ciA)
            fractionB_means_refs.append(meanB)
            fractionB_cis_refs.append(ciB)
        
            aggregated_fractions_refs[layer_name] = {
                "meanA": float(meanA),
                "ciA": float(ciA),
                "meanB": float(meanB),
                "ciB": float(ciB)
            }

        # Collect the means (order matters)
        layer_order = list(aggregated_fractions_refs.keys())
        
        meanA_vals = [aggregated_fractions_refs[layer]["meanA"] for layer in layer_order]
        meanB_vals = [aggregated_fractions_refs[layer]["meanB"] for layer in layer_order]

        print(meanA_vals)
        # Normalize using RobustScaler
        scalerA = RobustScaler()
        scalerB = RobustScaler()
        
        meanA_norm = scalerA.fit_transform(np.array(meanA_vals).reshape(-1, 1)).flatten()
        meanB_norm = scalerB.fit_transform(np.array(meanB_vals).reshape(-1, 1)).flatten()

        print(meanA_norm)
        
        # Write normalized values back into the dict
        for i, layer_name in enumerate(layer_order):
            aggregated_fractions_refs[layer_name]["meanA"] = float(meanA_norm[i])
            aggregated_fractions_refs[layer_name]["meanB"] = float(meanB_norm[i]) 
            
        # -- B) Text Pairs --
        # data["text_pairs"] is { pair_key: { layer_name: [(simA, simB), ...], ... } }
        aggregated_fractions_textpairs = {}
        for pair_key, layer_dict in data["text_pairs"].items():
            pair_out_dir = model_out_dir  # or a subfolder if you prefer
            layer_names_text = sorted(layer_dict.keys())
            fractionA_means_text = []
            fractionA_cis_text = []
            fractionB_means_text = []
            fractionB_cis_text = []

            aggregated_fractions_textpairs[pair_key] = {}

            for layer_name in layer_names_text:
                sim_list = layer_dict[layer_name]
                fractionA_vals = []
                fractionB_vals = []
                for (a, b) in sim_list:
                    s = a + b
                    if s > 1e-8:
                        fractionA_vals.append(a)
                        fractionB_vals.append(b)
                    else:
                        fractionA_vals.append(0.0)
                        fractionB_vals.append(0.0)

                meanA, ciA = confidence_interval(fractionA_vals, confidence=0.95)
                meanB, ciB = confidence_interval(fractionB_vals, confidence=0.95)

                fractionA_means_text.append(meanA)
                fractionA_cis_text.append(ciA)
                fractionB_means_text.append(meanB)
                fractionB_cis_text.append(ciB)

                aggregated_fractions_textpairs[pair_key][layer_name] = {
                    "meanA": meanA,
                    "ciA": ciA,
                    "meanB": meanB,
                    "ciB": ciB
                }

        # Normalize within each text pair
        for pair_key, layer_dict in aggregated_fractions_textpairs.items():
            layer_order = list(layer_dict.keys())
        
            # Extract meanA and meanB across layers
            meanA_vals = [layer_dict[layer]["meanA"] for layer in layer_order]
            meanB_vals = [layer_dict[layer]["meanB"] for layer in layer_order]
        
            # Apply RobustScaler
            scalerA = RobustScaler()
            scalerB = RobustScaler()
        
            meanA_norm = scalerA.fit_transform(np.array(meanA_vals).reshape(-1, 1)).flatten()
            meanB_norm = scalerB.fit_transform(np.array(meanB_vals).reshape(-1, 1)).flatten()
        
            # Write back normalized values
            for i, layer_name in enumerate(layer_order):
                layer_dict[layer_name]["meanA"] = float(meanA_norm[i])
                layer_dict[layer_name]["meanB"] = float(meanB_norm[i])


        # -- C) Save aggregated fraction data to JSON for this model
        out_json_path = os.path.join(model_out_dir, which_sim.split('.')[0] + "_aggregated_fraction_data_with_ci.json")
        aggregated_data = {
            "model": model_name,
            "count": model_data[model_name]["count"],
            "references": aggregated_fractions_refs,
            "text_pairs": aggregated_fractions_textpairs
        }
        with open(out_json_path, "w") as f:
            json.dump(aggregated_data, f, indent=4)

        print(f"Saved aggregated data & plots with CIs for model '{model_name}' in: {model_out_dir}")

if __name__ == "__main__":
    main("cv_similarities.json")
    #main("cv_similarities.json")
    #main("similarities.json")
