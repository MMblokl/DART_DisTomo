from source.algorithms import DART, SDART, SSIRT
from source.utils import create_sinogram
from source.metrics import calc_rnmp, calc_ssim
import glob
from PIL import Image
import numpy as np
import json
import sys
import astra


a_vals = [1,4,8,16]
try:
    clean_or_noisy = sys.argv[1]
except IndexError:
    noisy = False
else:
    if "noisy" in clean_or_noisy or "noise" in clean_or_noisy:
        noisy = True
    else:
        noisy = False

try:
    seed_choice = sys.argv[2]
    seed = int(seed_choice)
except (IndexError, ValueError):
    seed = 202667
print(f"Running with noisy: {noisy} and seed: {seed}")

np.random.seed(seed)

final_results = {
        "blob": {
            1: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            4: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            8: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            16: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}}
        },
        "bone": {
            1: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            4: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            8: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            16: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}}
        },
        "mesh": {
            1: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            4: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            8: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}},
            16: {"ssirt": {"rnmp": 0, "ssim": 0}, "dart": {"rnmp": 0, "ssim": 0}, "sdart": {"rnmp": 0, "ssim": 0}}
        },
    }
results = {
        "blob": {
            1: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            4: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            8: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            16: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}}
        },
        "bone": {
            1: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            4: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            8: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            16: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}}
        },
        "mesh": {
            1: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            4: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            8: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}},
            16: {"ssirt": {"rnmp": [], "ssim": []}, "dart": {"rnmp": [], "ssim": []}, "sdart": {"rnmp": [], "ssim": []}}
        },
    }
for phantom_group in glob.glob("./phantoms/*"):
    for phantom in glob.glob(f"{phantom_group}/*.png"):

        if "blob" in phantom:
            grey_intensities = [0,120,255]
            p_group = "blob"
        elif "bone" in phantom:
            grey_intensities = [0, 110, 150, 220]
            p_group = "bone"
        else:
            grey_intensities = [0,255]
            p_group = "mesh"
        img = Image.open(phantom)
        img = np.asarray(img)
    
        proj_geom, sino = create_sinogram(img, 64, 180)
        
        # Add noise if noisy is active
        if noisy:
            sino = astra.functions.add_noise_to_sino(sino, 1e5, seed=seed)

        # Run each algorithm on this phantom using different a supersampling values for the reconstruction
        for a_val in a_vals:
            ssirt = SSIRT.SSIRT(
                proj_geom=proj_geom,
                sinogram=sino,
                img_shape=img.shape,
                supersampling_a=a_val
            )
            dart = DART.DART(
                proj_geom=proj_geom,
                sinogram=sino,
                img_shape=img.shape,
                supersampling_a=a_val,
                reconstruction_iterations=100,
            )
            sdart = SDART.SDART(
                proj_geom=proj_geom,
                sinogram=sino,
                img_shape=img.shape,
                supersampling_a=a_val,
                lambda_hp=0.1,
                reconstruction_iterations=100,
            )

            # Reconstruction for this a-value
            ssirt_res = ssirt.run(grey_intensities=grey_intensities, iterations=100)
            dart_res = dart.run(p=0.4, grey_intensities=grey_intensities, iterations=100)
            sdart_res = sdart.run(grey_intensities=grey_intensities, iterations=100)

            # Save results
            results[p_group][a_val]["ssirt"]["rnmp"].append(calc_rnmp(img, ssirt_res))
            results[p_group][a_val]["ssirt"]["ssim"].append(calc_ssim(img, ssirt_res))
            results[p_group][a_val]["dart"]["rnmp"].append(calc_rnmp(img, dart_res))
            results[p_group][a_val]["dart"]["ssim"].append(calc_ssim(img, dart_res))
            results[p_group][a_val]["sdart"]["rnmp"].append(calc_rnmp(img, sdart_res))
            results[p_group][a_val]["sdart"]["ssim"].append(calc_ssim(img, sdart_res))
        
        print(f"Finished phantom {phantom}.")
    print(f"Finished famility {phantom_group}.")

# Average results over each phantom group
for phantom_group in results.keys():
    for a_val in results[phantom_group].keys():
        for recon_alg in results[phantom_group][a_val].keys():
            final_results[phantom_group][a_val][recon_alg]["rnmp"] = np.mean(results[phantom_group][a_val][recon_alg]["rnmp"])
            final_results[phantom_group][a_val][recon_alg]["ssim"] = np.mean(results[phantom_group][a_val][recon_alg]["ssim"])


# Save results to JSON
filename = "./noisy_results.json" if noisy else "./clean_results.json"
with open(filename, "w") as f:
    json.dump(final_results, f) 
