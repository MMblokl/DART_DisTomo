from source.algorithms.DART import DART
from source.sinograms.create_sinogram import create_sinogram
from source.metrics import calc_rnmp, calc_ssim
from PIL import Image
import numpy as np
import glob
import json

ps = [0.1, 0.2, 0.4, 0.8]
phantoms = ["blob", "bone", "mesh"]
detector_numbers = [512, 128, 64]

# Test DART
results = {
    phantom: {
        p_val: {
            det: {"rnmp": [], "ssim": []} for det in detector_numbers
            } for p_val in ps
    } for phantom in phantoms
}
final_results = {
    phantom: {
        p_val: {
            det: {"rnmp": 0, "ssim": 0} for det in detector_numbers
            } for p_val in ps
    } for phantom in phantoms
}

breakpoint()
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

        for p_val in ps:
            for n_detectors in detector_numbers:
                proj_geom, sino = create_sinogram(img, n_detectors, 180)

                dart = DART(
                    proj_geom=proj_geom,
                    sinogram=sino,
                    img_shape=img.shape,
                    supersampling_a=1,
                    reconstruction_iterations=100,
                )
            
                rec_img = dart.run(p=p_val, gray_intensities=grey_intensities, iterations=100)
                rnmp = calc_rnmp(img, rec_img)
                ssim = calc_ssim(img, rec_img)

                results[p_group][p_val][n_detectors]["rnmp"].append(rnmp)
                results[p_group][p_val][n_detectors]["ssim"].append(ssim)
    
        print(f"Finished with phantom {phantom}.")
    print(f"Finished with phantom family {phantom_group}.")


for phantom_group in results.keys():
    for p_val in results[phantom].keys():
        for n_detectors in results[phantom_group][p_val].keys():
            final_results[phantom_group][p_val][n_detectors]["rnmp"] = np.mean(results[phantom_group][p_val][n_detectors]["rnmp"])
            final_results[phantom_group][p_val][n_detectors]["ssim"] = np.mean(results[phantom_group][p_val][n_detectors]["ssim"])


with open("p_dart_results.json", "w") as f:
    json.dump(final_results, f)
