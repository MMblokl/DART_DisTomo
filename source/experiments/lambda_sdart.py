from source.algorithms.SDART import SDART
from source.sinograms.create_sinogram import create_sinogram
from source.metrics import calc_rnmp, calc_ssim
from PIL import Image
import numpy as np
import glob
import json

lambdas = [0.1, 0.24, 0.48, 0.8]
phantoms = ["blob", "bone", "mesh"]
detector_numbers = [512, 128, 64]

# Test SDART
results = {
    phantom: {
        lambda_val: {
            det: {"rnmp": [], "ssim": []} for det in detector_numbers
            } for lambda_val in lambdas
    } for phantom in phantoms
}
final_results = {
    phantom: {
        lambda_val: {
            det: {"rnmp": 0, "ssim": 0} for det in detector_numbers
            } for lambda_val in lambdas
    } for phantom in phantoms
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

        for lambda_val in lambdas:
            for n_detectors in detector_numbers:
                proj_geom, sino = create_sinogram(img, n_detectors, 180)

                dart = SDART(
                    proj_geom=proj_geom,
                    sinogram=sino,
                    img_shape=img.shape,
                    supersampling_a=1,
                    reconstruction_iterations=100,
                    lambda_hp=lambda_val
                )
            
                rec_img = dart.run(gray_intensities=grey_intensities, iterations=100)
                rnmp = calc_rnmp(img, rec_img)
                ssim = calc_ssim(img, rec_img)

                results[p_group][lambda_val][n_detectors]["rnmp"].append(rnmp)
                results[p_group][lambda_val][n_detectors]["ssim"].append(ssim)
    
        print(f"Finished with phantom {phantom}.")
    print(f"Finished with phantom family {phantom_group}.")


for phantom_group in results.keys():
    for lambda_val in results[phantom].keys():
        for n_detectors in results[phantom_group][lambda_val].keys():
            final_results[phantom_group][lambda_val][n_detectors]["rnmp"] = np.mean(results[phantom_group][lambda_val][n_detectors]["rnmp"])
            final_results[phantom_group][lambda_val][n_detectors]["ssim"] = np.mean(results[phantom_group][lambda_val][n_detectors]["ssim"])


with open("lambda_results.json", "w") as f:
    json.dump(final_results, f)
