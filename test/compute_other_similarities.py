import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.feature import local_binary_pattern
from tqdm import tqdm 

def plot_stacked_bar(layer_names,
                     simA_list,
                     simB_list,
                     title,
                     output_path,
                     legend_A="Similarity with Ref A",
                     legend_B="Similarity with Ref B",
                     normalize_sum_to_1=True):
    """
    Creates a stacked bar plot with:
      - X-axis: layer_names
      - Y-axis: simA + simB (or normalized to 1 if normalize_sum_to_1=True)
      - Lower (orange) bar: simA
      - Upper (blue) bar: simB
    """
    if normalize_sum_to_1:
        normA = []
        normB = []
        for a, b in zip(simA_list, simB_list):
            s = a + b
            if s > 0:
                normA.append(a / s)
                normB.append(b / s)
            else:
                normA.append(0.0)
                normB.append(0.0)
        simA_list = normA
        simB_list = normB

    x_positions = np.arange(len(layer_names))
    simA_array = np.array(simA_list)
    simB_array = np.array(simB_list)

    plt.figure(figsize=(10, 5))
    plt.bar(x_positions, simA_array, color='orange', label=legend_A)
    plt.bar(x_positions, simB_array, bottom=simA_array, color='blue', label=legend_B)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Similarity" if not normalize_sum_to_1 else "Proportion of Similarity")
    plt.xticks(x_positions, layer_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compute_cv_similarities(image1_path, image2_path):
    """
    Computes four similarity metrics between two images:
      1. Structural similarity (SSIM)
      2. Keypoint similarity (SIFT/ORB-based)
      3. Color histogram similarity (in HSV space)
      4. Texture similarity (using Local Binary Patterns)
      
    Returns a dictionary with keys: "ssim", "keypoint", "color", "texture"
    """
    # Load images using OpenCV
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None or img2 is None:
        raise ValueError("One of the images could not be loaded.")

    # Convert to grayscale (for SSIM and keypoints)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 1. Structural Similarity (SSIM)
    ssim_score, _ = compare_ssim(gray1, gray2, full=True)
    
    # 2. Keypoint Similarity using SIFT (fallback to ORB if unavailable)
    try:
        sift = cv2.SIFT_create()
    except Exception:
        sift = cv2.ORB_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    keypoint_score = 0.0
    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        if hasattr(sift, 'descriptorSize'):  # SIFT uses Euclidean distance
            bf = cv2.BFMatcher()
        else:  # ORB uses Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        keypoint_score = len(good_matches) / max(len(kp1), len(kp2))
    
    # 3. Color Similarity using HSV Histograms
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    color_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 4. Texture Similarity using Local Binary Patterns (LBP)
    lbp1 = local_binary_pattern(gray1, 8, 1, method="uniform")
    lbp2 = local_binary_pattern(gray2, 8, 1, method="uniform")
    n_bins = int(lbp1.max() + 1)
    hist_lbp1, _ = np.histogram(lbp1.ravel(), bins=n_bins, range=(0, n_bins))
    hist_lbp2, _ = np.histogram(lbp2.ravel(), bins=n_bins, range=(0, n_bins))
    hist_lbp1 = hist_lbp1.astype("float")
    hist_lbp2 = hist_lbp2.astype("float")
    hist_lbp1 /= (hist_lbp1.sum() + 1e-7)
    hist_lbp2 /= (hist_lbp2.sum() + 1e-7)
    texture_score = cv2.compareHist(hist_lbp1.astype(np.float32), hist_lbp2.astype(np.float32), cv2.HISTCMP_CORREL)
    
    return {
        "ssim": ssim_score,
        "keypoint": keypoint_score,
        "color": color_score,
        "texture": texture_score
    }

def main():
    # Top-level outputs folder
    outputs_dir = "../outputs_test_2"

    # Process each subfolder containing the experiment data
    for subfolder in tqdm(os.listdir(outputs_dir)):
            try:
                subfolder_path = os.path.join(outputs_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                out_json_path = os.path.join(subfolder_path, "cv_similarities.json")
                if os.path.isfile(out_json_path):
                    print('Folder already processed:', out_json_path)
                    continue
                    
                # Optional: read configuration (if present)
                config_path = os.path.join(subfolder_path, "config.json")
                if os.path.isfile(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                else:
                    config = {}

                # Locate reference images (e.g., output_A and output_C)
                output_A_path = None
                output_B_path = None
                injected_image_paths = []

                for fname in os.listdir(subfolder_path):
                    if fname.startswith("A"):
                        output_A_path = os.path.join(subfolder_path, fname)
                    elif fname.startswith("B"):
                        output_B_path = os.path.join(subfolder_path, fname)
                    elif fname.startswith("C"):
                        injected_image_paths.append(os.path.join(subfolder_path, fname))

                if not output_A_path or not output_B_path:
                    print(f"Skipping {subfolder} because reference images not found.")
                    continue

                # Compute baseline similarities between the two reference images
                baseline_similarities = compute_cv_similarities(output_A_path, output_B_path)
                #print(f"Baseline OpenCV similarities in {subfolder}:")
                #print(baseline_similarities)

                model_name = config.get("model", "unknown_model")
                prompt_A = config.get("prompt_A", "")
                prompt_B = config.get("prompt_B", "")

                # Prepare to store results
                similarities_dict = {
                    "model": model_name,
                    "prompt_A": prompt_A,
                    "prompt_B": prompt_B,
                    "cv_similarities": {},  # For sim to output_A & output_B
                }
                
                # For plotting, we accumulate scores per metric per injected image (layer)
                metric_sims = {
                    "ssim": {},
                    "keypoint": {},
                    "color": {},
                    "texture": {}
                }

                # Compute similarities for each injected image (assumed to correspond to a layer)
                for img_path in injected_image_paths:
                    #print(len(injected_image_paths), img_path)
                    fname = os.path.basename(img_path)
                    # Extract layer name (e.g., for "injected_unet.down_blocks.2.resnets.1.png")
                    layer_name = fname.replace("C_skips_", "").replace(".png", "") #fname.split("_")[-1].replace(".png", "")
                    # Compute each metric separately against both reference images
                    simA_metrics = compute_cv_similarities(img_path, output_A_path)
                    simB_metrics = compute_cv_similarities(img_path, output_B_path)
                    
                    similarities_dict["cv_similarities"][layer_name] = {
                        "ssim": {"simA": max(0, simA_metrics["ssim"] - baseline_similarities["ssim"]), "simB": max(0, simB_metrics["ssim"] - baseline_similarities["ssim"])},
                        "keypoint": {"simA": simA_metrics["keypoint"], "simB": simB_metrics["keypoint"]},
                        "color": {"simA": max(0, simA_metrics["color"] - baseline_similarities["color"]), "simB": max(0, simB_metrics["color"] - baseline_similarities["color"])},
                        "texture": {"simA": max(0, simA_metrics["texture"] - baseline_similarities["texture"]), "simB": max(0, simB_metrics["texture"] - baseline_similarities["texture"])}
                    }
                    
                    # Save for plotting
                    for metric in ["ssim", "keypoint", "color", "texture"]:
                        metric_sims[metric][layer_name] = (
                            similarities_dict["cv_similarities"][layer_name][metric]["simA"],
                            similarities_dict["cv_similarities"][layer_name][metric]["simB"]
                        )

                # Save the similarities into a JSON file
                with open(out_json_path, "w") as f:
                    json.dump(similarities_dict, f, indent=4)

                # Determine a sorted order for the layers (injected images)
                #layer_order = sorted(metric_sims["ssim"].keys())

                # For each metric, create and save a separate stacked bar plot
                #for metric in ["ssim", "keypoint", "color", "texture"]:
                #    simA_values = [metric_sims[metric][layer][0] for layer in layer_order]
                #    simB_values = [metric_sims[metric][layer][1] for layer in layer_order]
                #    bar_title = f"OpenCV {metric.capitalize()} Similarities\n{subfolder}"
                #    bar_path = os.path.join(subfolder_path, f"cv_similarities_bar_{metric}.png")
                #    plot_stacked_bar(layer_order, simA_values, simB_values,
                #                    title=bar_title,
                #                    output_path=bar_path,
                #                    legend_A="Similarity with Ref A",
                #                    legend_B="Similarity with Ref B")
                
                print(f"Processed {subfolder}. Saved JSON and bar plots in {subfolder_path}.")

            except Exception as e:
                print(e)
                print(subfolder)
                
if __name__ == "__main__":
    main()
