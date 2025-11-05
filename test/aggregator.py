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

def plot_stacked_bar_with_conjunction_ci(layer_names,
                                         fractionA_means,
                                         fractionA_cis,
                                         fractionB_means,
                                         fractionB_cis,
                                         title,
                                         output_path,
                                         legend_A="Fraction A",
                                         legend_B="Fraction B"):
    """
    Creates a stacked bar chart with CI bars at the conjunction point between A and B:
    - Lower CI for fraction A
    - Upper CI for fraction B
    """
    x_positions = np.arange(len(layer_names))

    # Compute confidence intervals centered at the transition
    conjunction_means = np.array(fractionA_means)  # Boundary between A and B
    lower_errors = np.array(fractionA_cis)  # Lower half CI for A
    upper_errors = np.array(fractionB_cis)  # Upper half CI for B

    plt.figure(figsize=(10, 5))

    # Bottom part (Fraction A)
    plt.bar(x_positions, fractionA_means, color='orange', label=legend_A)

    # Top part (Fraction B), stacked on Fraction A
    plt.bar(x_positions, fractionB_means, bottom=fractionA_means, color='blue', label=legend_B)

    # Add error bars centered at the conjunction between A and B
    plt.errorbar(x_positions, conjunction_means, yerr=[lower_errors, upper_errors],
                 fmt='none', ecolor='black', capsize=3, label="CI at conjunction")

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Fraction of Similarity (A vs B)")
    plt.xticks(x_positions, layer_names, rotation=45, ha="right")
    plt.ylim([0, 1.05])  # Keep within range
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(which_sim):
    outputs_dir = "output_ours_35_exp"

    model_data = {}

    # 1) Collect data from each subfolder's similarities.json
    for subfolder in os.listdir(outputs_dir):
        subfolder_path = os.path.join(outputs_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        similarities_path = os.path.join(subfolder_path, which_sim)
        if not os.path.isfile(similarities_path):
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
        text_sims = similarities.get("cv_similarities", {})
        for layer_name, pair_dict in text_sims.items():
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
            # Compute fraction arrays for each run
            fractionA_vals = []
            fractionB_vals = []
            for (a, b) in sim_list:
                s = a + b
                if s > 1e-8:
                    fractionA_vals.append(a / s)
                    fractionB_vals.append(b / s)
                else:
                    fractionA_vals.append(0.0)
                    fractionB_vals.append(0.0)

            meanA, ciA = confidence_interval(fractionA_vals, confidence=0.90)
            meanB, ciB = confidence_interval(fractionB_vals, confidence=0.90)

            fractionA_means_refs.append(meanA)
            fractionA_cis_refs.append(ciA)
            fractionB_means_refs.append(meanB)
            fractionB_cis_refs.append(ciB)

            aggregated_fractions_refs[layer_name] = {
                "meanA": meanA,
                "ciA": ciA,
                "meanB": meanB,
                "ciB": ciB
            }

        # Plot stacked bar with error bars for references
        ref_plot_title = f"Aggregated Fractions w/ CIs (References)\nModel: {model_name}"
        ref_plot_path = os.path.join(model_out_dir, "avg_stacked_refs_with_ci.png")
        plot_stacked_bar_with_conjunction_ci(
            layer_names_refs,
            fractionA_means_refs,
            fractionA_cis_refs,
            fractionB_means_refs,
            fractionB_cis_refs,
            title=ref_plot_title,
            output_path=ref_plot_path,
            legend_A="Fraction w/ output_A",
            legend_B="Fraction w/ output_B"
        )

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
                        fractionA_vals.append(a / s)
                        fractionB_vals.append(b / s)
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

            # Plot stacked bar with error bars for this text pair
            pair_plot_title = f"Aggregated Fractions w/ CIs (Text Pair: {pair_key})\nModel: {model_name}"
            safe_pair_key = pair_key.replace(" ", "_").replace("/", "-")
            pair_plot_path = os.path.join(pair_out_dir, f"avg_stacked_{safe_pair_key}_with_ci.png")
            # Derive text labels for the legend from the "X vs Y" pattern, if present
            if " vs " in pair_key:
                legendA = f"Fraction w/ \"{pair_key.split(' vs ')[0]}\""
                legendB = f"Fraction w/ \"{pair_key.split(' vs ')[1]}\""
            else:
                legendA = "Fraction A"
                legendB = "Fraction B"

            plot_stacked_bar_with_conjunction_ci(
                layer_names_text,
                fractionA_means_text,
                fractionA_cis_text,
                fractionB_means_text,
                fractionB_cis_text,
                title=pair_plot_title,
                output_path=pair_plot_path,
                legend_A=legendA,
                legend_B=legendB
            )

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
    main("similarities.json")
