import os
import json
import numpy as np
import scipy.stats


# ======================================================
# Utils
# ======================================================

def confidence_interval(data, confidence=0.95):
    a = np.asarray(data, dtype=float)
    n = len(a)
    if n < 2:
        return float(np.mean(a)), 0.0

    mean = np.mean(a)
    sem = scipy.stats.sem(a)
    t_val = scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(mean), float(sem * t_val)


def quantile_normalizer(reference):
    """
    Returns a callable ECDF-based normalizer mapping to [0, 1].
    """
    reference = np.asarray(reference, dtype=float)

    if reference.size == 0:
        return lambda x: np.zeros_like(x, dtype=float)

    if reference.size == 1:
        return lambda x: np.zeros_like(x, dtype=float)

    ref_sorted = np.sort(reference)
    grid = np.linspace(0, 1, len(ref_sorted))

    def transform(x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, ref_sorted, grid, left=0.0, right=1.0)

    return transform


# ======================================================
# JSON normalization
# ======================================================

def extract_similarities(payload, filename):
    """
    Normalize both:
    - cv_similarities.json (old)
    - results.json (new)
    into a canonical structure:
      refs[layer] -> [(simA, simB), ...]
      text_pairs[pair_key][layer] -> [(simA, simB), ...]
    """

    refs = {}
    text_pairs = {}

    # ---- OLD FORMAT ----
    if "image_similarities" in payload or "text_similarities" in payload:
        for layer, v in payload.get("image_similarities", {}).items():
            refs.setdefault(layer, []).append(
                (v.get("simA", 0.0), v.get("simB", 0.0))
            )

        for layer, pairs in payload.get("text_similarities", {}).items():
            for pair_key, v in pairs.items():
                text_pairs.setdefault(pair_key, {}).setdefault(layer, []).append(
                    (v.get("simA", 0.0), v.get("simB", 0.0))
                )

        key = os.path.splitext(filename)[0]
        if key in payload:
            for layer, pairs in payload[key].items():
                for pair_key, v in pairs.items():
                    simA = v.get("simA", 0.0)
                    simB = v.get("simB", 0.0)
                    refs.setdefault(layer, []).append((simA, simB))
                    text_pairs.setdefault(pair_key, {}).setdefault(layer, []).append(
                        (simA, simB)
                    )

    # ---- NEW FORMAT (results.json) ----
    else:
        for layer, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue

            for metric, v in metrics.items():
                if "_" in metric:
                    simB = float(v)
                    simA = 1.0 - simB
                    refs.setdefault(layer, []).append((simA, simB))
                    text_pairs.setdefault(metric, {}).setdefault(layer, []).append(
                        (simA, simB)
                    )

    return refs, text_pairs


# ======================================================
# Main
# ======================================================

def main(sim_filename):

    model = 'outputs_stable-diffusion-v1-4_controlled'
    outputs_dir = f"outputs_per_model/{model}"
    llm = True

    model_data = {}

    # ------------------
    # Load all runs
    # ------------------
    for run in os.listdir(outputs_dir):
        run_path = os.path.join(outputs_dir, run)
        if not os.path.isdir(run_path):
            continue

        json_path = os.path.join(run_path, sim_filename)
        if not os.path.isfile(json_path):
            continue

        with open(json_path) as f:
            payload = json.load(f)

        model_name = outputs_dir.split('/')[-1]
        model_data.setdefault(model_name, {
            "refs": {},
            "text_pairs": {},
            "count": 0
        })

        refs, text_pairs = extract_similarities(payload, sim_filename)

        for layer, vals in refs.items():
            model_data[model_name]["refs"].setdefault(layer, []).extend(vals)

        for pair_key, layers in text_pairs.items():
            for layer, vals in layers.items():
                model_data[model_name]["text_pairs"] \
                    .setdefault(pair_key, {}) \
                    .setdefault(layer, []).extend(vals)

        model_data[model_name]["count"] += 1

    # ------------------
    # Aggregation
    # ------------------
    for model, data in model_data.items():
        out_dir = os.path.join(outputs_dir, "averages", model)
        os.makedirs(out_dir, exist_ok=True)

        # ===== References =====
        refs_out = {}
        for layer, sims in data["refs"].items():
            A = [a for a, _ in sims]
            B = [b for _, b in sims]

            mA, cA = confidence_interval(A, 0.9)
            mB, cB = confidence_interval(B, 0.9)

            refs_out[layer] = dict(meanA=mA, ciA=cA, meanB=mB, ciB=cB)

        layers = list(refs_out.keys())
        if layers:
            meanA_vals = np.array([refs_out[l]["meanA"] for l in layers])
            meanB_vals = np.array([refs_out[l]["meanB"] for l in layers])

            qA = quantile_normalizer(meanA_vals) 
            qB = quantile_normalizer(meanB_vals)

            for l in layers:
                mA, cA = refs_out[l]["meanA"], refs_out[l]["ciA"]
                mB, cB = refs_out[l]["meanB"], refs_out[l]["ciB"]

                loA, hiA = qA([mA - cA, mA + cA])
                loB, hiB = qB([mB - cB, mB + cB])

                refs_out[l]["meanA"] = float(qA([mA])[0]) if not llm else mA / 10
                refs_out[l]["ciA"]   = float((hiA - loA) / 2)

                refs_out[l]["meanB"] = float(qB([mB])[0]) if not llm else mB / 10
                refs_out[l]["ciB"]   = float((hiB - loB) / 2)

        # ===== Text pairs =====
        text_out = {}
        for pair, layers_dict in data["text_pairs"].items():
            text_out[pair] = {}

            for layer, sims in layers_dict.items():
                A = [a for a, _ in sims]
                B = [b for _, b in sims]

                mA, cA = confidence_interval(A)
                mB, cB = confidence_interval(B)

                text_out[pair][layer] = dict(
                    meanA=mA, ciA=cA,
                    meanB=mB, ciB=cB
                )

            layer_names = list(text_out[pair].keys())
            if not layer_names:
                continue

            meanA_vals = np.array([text_out[pair][l]["meanA"] for l in layer_names])
            meanB_vals = np.array([text_out[pair][l]["meanB"] for l in layer_names])

            qA = quantile_normalizer(meanA_vals)
            qB = quantile_normalizer(meanB_vals)

            for l in layer_names:
                mA, cA = text_out[pair][l]["meanA"], text_out[pair][l]["ciA"]
                mB, cB = text_out[pair][l]["meanB"], text_out[pair][l]["ciB"]

                loA, hiA = qA([mA - cA, mA + cA])
                loB, hiB = qB([mB - cB, mB + cB])

                text_out[pair][l]["meanA"] = float(qA([mA])[0]) if not llm else mA / 10
                text_out[pair][l]["ciA"]   = float((hiA - loA) / 2)

                text_out[pair][l]["meanB"] = float(qB([mB])[0]) if not llm else mB / 10
                text_out[pair][l]["ciB"]   = float((hiB - loB) / 2)

        # ===== Save =====
        out_json = {
            "model": model,
            "count": data["count"],
            "references": refs_out,
            "text_pairs": text_out
        }

        out_path = os.path.join(
            out_dir,
            os.path.splitext(sim_filename)[0]
            + "_aggregated_fraction_data_with_ci.json"
        )

        with open(out_path, "w") as f:
            json.dump(out_json, f, indent=2)

        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main("results.json")          # new
    # main("cv_similarities.json") # old
