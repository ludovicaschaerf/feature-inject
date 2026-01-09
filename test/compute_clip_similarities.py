import os
import json
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a CLIP model from Hugging Face
model_id = "openai/clip-vit-base-patch32" 
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def compute_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
    # Normalize
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds

def compute_text_embeddings(text_list):
    inputs = processor(text=text_list, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
    # Normalize
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds

def main():
    #models = [
    #    "outputs_kandinsky-2-2-decoder_controlled",
    #    "outputs_stable-diffusion-2-base_controlled",
    #    "outputs_sd-turbo_controlled",
    #    "outputs_stable-diffusion-v1-4_controlled",
    #    "outputs_sdxl-turbo_controlled",
    #    "outputs_stable-diffusion-xl-base-1.0_controlled",
    #]

    models = ["outputs_FLUX.1-schnell_controlled", 
              "outputs_PixArt-XL-2-1024-MS_controlled",
              "outputs_stable-diffusion-3.5-large-turbo_controlled"]
    
    for model in models:
        print("Processing:", model)
        outputs_dir = f"../outputs_per_model/{model}"

        for subfolder in tqdm(os.listdir(outputs_dir)[:100]):
            try:
                subfolder_path = os.path.join(outputs_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    print('Not a folder, skipping')
                    continue
    
                out_json_path = os.path.join(subfolder_path, "clip_similarities.json")
                if os.path.isfile(out_json_path):
                    print('Folder already processed:', out_json_path)
                    continue
                    
                # Look for config.json
                config_path = os.path.join(subfolder_path, "config.json")
                if not os.path.isfile(config_path):
                    config = {}
                else:
                    # Read config
                    with open(config_path, "r") as f:
                        config = json.load(f)
    
                # Extract some info from config for reference
                model_name = config.get("model", "unknown_model")
                prompt_A = config.get("prompt_A", "")
                prompt_B = config.get("prompt_B", "")
    
               
                # Locate reference images (output_A*, output_B*)
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
    
                # If we don't have references, skip
                if not output_A_path or not output_B_path:
                    print(f"Skipping {subfolder} because reference images not found.")
                    continue
    
                # Compute CLIP embeddings for reference images
                embedding_A = compute_image_embedding(output_A_path)
                embedding_B = compute_image_embedding(output_B_path)
    
                original_difference = float((embedding_B * embedding_A).sum().item())
                    
                # Prepare to store results
                similarities_dict = {
                        "model": model_name,
                        "prompt_A": prompt_A,
                        "prompt_B": prompt_B,
                        "image_similarities": {},  # For sim to output_A & output_B
                        "text_similarities": {}    # For sim to text pairs
                }
    
                # For histograms of references
                simA_values = []
                simB_values = []
                layer_order = []  # Keep track of layer order as we encounter them
    
                    
                # We'll store per-layer reference sims in a dictionary
                # layer_sims_refs[layer_name] = (simA, simB)
                layer_sims_refs = {}
    
                # Compute similarity for each injected image
                for img_path in injected_image_paths:
                    fname = os.path.basename(img_path)
                    # Extract layer name (e.g., for "injected_unet.down_blocks.2.resnets.1.png")
                    layer_name = fname.replace("C_skips_", "").replace(".png", "") #fname.split("_")[-1].replace(".png", "")
                    
                    embedding_injected = compute_image_embedding(img_path) 
                        
                    # Cosine similarity with reference images A and B
                    simA = max(0, float((embedding_injected * embedding_A).sum().item()) - original_difference)
                    simB = max(0, float((embedding_injected * embedding_B).sum().item()) - original_difference)
    
                    # Store in the dictionary
                    similarities_dict["image_similarities"].setdefault(layer_name, {})
                    similarities_dict["image_similarities"][layer_name]["simA"] = simA
                    similarities_dict["image_similarities"][layer_name]["simB"] = simB
    
                    simA_values.append(simA)
                    simB_values.append(simB)
                        
                    # Maintain layer order
                    if layer_name not in layer_order:
                        layer_order.append(layer_name)
    
                    # Also store for bar plotting
                    layer_sims_refs[layer_name] = (simA, simB)
    
    
                layer_order.sort()
                    
                # Save similarities to JSON
                out_json_path = os.path.join(subfolder_path, "clip_similarities.json")
                with open(out_json_path, "w") as f:
                    json.dump(similarities_dict, f, indent=4)
    
                print(f"Processed {subfolder}. Saved JSON and bar plots in {subfolder_path}.")


            except Exception as e:
                print(e)
                print(subfolder_path)
                                
            
if __name__ == "__main__":
    main()
