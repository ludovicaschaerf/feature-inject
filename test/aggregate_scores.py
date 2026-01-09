import os
import json
import numpy as np
import scipy.stats


# ======================================================
# Utils
# ======================================================

def confidence_interval(
    data,
    confidence=0.95,
    n_resamples=10000,
    random_state=None,
    method="BCa",
):
    import matplotlib.pyplot as plt

    a = np.asarray(data, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size

    mean = float(np.median(a)) if n > 0 else 0

    if n < 3:
        return mean, 0.0

    res = scipy.stats.bootstrap(
        (a,),
        statistic=np.median,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
        random_state=random_state,
        vectorized=False,
    )

    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)
    half_width = (high - low) / 2.0

    return mean, half_width


def quantile_normalizer(reference): 
    """ Returns a callable ECDF-based normalizer mapping to [0, 1]. """ 
    reference = np.asarray(reference, dtype=float) 
    if reference.size == 0 or reference.size == 1: 
        return lambda x: np.zeros_like(x, dtype=float) 
    ref_sorted = np.sort(reference) 
    grid = np.linspace(0, 1, len(ref_sorted)) 
    def transform(x): 
        x = np.asarray(x, dtype=float) 
        return np.interp(x, ref_sorted, grid, left=0.0, right=1.0) 
    return transform 


def extract_similarities(payload):
    """
    Normalize both:
    - legacy cv_similarities.json
    - newer results.json-style formats

    Returns:
      refs[layer] -> [(simA, simB), ...]
      text_pairs[pair_key][layer] -> [(simA, simB), ...]
    """

    refs = {}
    text_pairs = {}

    legacy_keys = [k for k in payload if "similarities" in k]
    if legacy_keys:
        k = legacy_keys[0]
        for layer, pairs in payload[k].items():
            for pair_key, v in pairs.items():
                simA = float(v.get("simA", 0.0))
                simB = float(v.get("simB", 0.0))

                refs.setdefault(layer, []).append((simA, simB))
                text_pairs.setdefault(pair_key, {}).setdefault(layer, []).append(
                    (simA, simB)
                )
        return refs, text_pairs

    for layer, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue

        for metric, v in metrics.items():
            if "_" not in metric:
                continue

            simB = float(v) / 10
            simA = 1.0 - simB

            refs.setdefault(layer, []).append((simA, simB))
            text_pairs.setdefault(metric, {}).setdefault(layer, []).append(
                (simA, simB)
            )

    return refs, text_pairs

def main(sim_filename, model, normal, outputs_dir): 
    model_data = {} 
    
    # ------------------ # Load all runs # ------------------ 
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
        model_data.setdefault(model_name, { "refs": {}, "text_pairs": {}, "count": 0 }) 
        
        refs, text_pairs = extract_similarities(payload) 
        for layer, vals in refs.items(): 
            model_data[model_name]["refs"].setdefault(layer, []).extend(vals) 
        
        for pair_key, layers in text_pairs.items(): 
            for layer, vals in layers.items(): 
                model_data[model_name]["text_pairs"] \
                    .setdefault(pair_key, {}) \
                    .setdefault(layer, []).extend(vals) 
        
        model_data[model_name]["count"] += 1 
        
    # ------------------ # Aggregation # ------------------ 
    for model, data in model_data.items(): 
        out_dir = os.path.join(outputs_dir, "averages", model) 
        os.makedirs(out_dir, exist_ok=True) 
    
    # ===== Text pairs ===== 
    text_out = {} 
    for pair, layers_dict in data["text_pairs"].items(): 
        text_out[pair] = {} 
        for layer, sims in layers_dict.items(): 
            A = [a for a, _ in sims] 
            B = [b for _, b in sims] 
            mA, cA = confidence_interval(A) 
            mB, cB = confidence_interval(B) 
            
            text_out[pair][layer] = dict( meanA=mA, ciA=cA, meanB=mB, ciB=cB ) 
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
                text_out[pair][l]["meanA"] = float(qA([mA])[0]) if normal else mA 
                text_out[pair][l]["ciA"] = float((hiA - loA) / 2) 
                text_out[pair][l]["meanB"] = float(qB([mB])[0]) if normal else mB 
                text_out[pair][l]["ciB"] = float((hiB - loB) / 2) 
    
    # ===== Save ===== 
    out_json = { "model": model, "count": data["count"], "text_pairs": text_out, } 
    out_path = os.path.join( 
        out_dir, os.path.splitext(sim_filename)[0] + "_aggregated_fraction_data_with_ci.json" 
    ) 
    with open(out_path, "w") as f: 
        json.dump(out_json, f, indent=2) 
    
    print(f"Saved → {out_path}") 
    
if __name__ == "__main__": 
    models = ["outputs_kandinsky-2-2-decoder_controlled", "outputs_stable-diffusion-2-base_controlled", "outputs_sd-turbo_controlled", "outputs_stable-diffusion-v1-4_controlled", "outputs_sdxl-turbo_controlled", "outputs_stable-diffusion-xl-base-1.0_controlled"]
    
    models = ["outputs_FLUX.1-schnell_controlled", "outputs_stable-diffusion-3.5-large-turbo_controlled"] 
    
    for model in models: 
        print('Processing', model) 
        outputs_dir = f"../outputs_per_model/{model}" 
        normal = False 
        sim_file = "cv_new_similarities.json" #results 
        main(sim_file, model, normal, outputs_dir)
        