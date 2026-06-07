from source.algorithms.DART import DART
from source.sinograms.create_sinogram import create_sinogram
from source.metrics import calc_rnmp, calc_ssim
from PIL import Image
import numpy as np
import glob
import json

ps = [0.1, 0.2, 0.4, 0.8]
phantoms = ["blob", "bone", "mesh"]

# Test DART
resuts = {
    phantom: {
        p_val: {"rnmp": [], "ssim": []} for p_val in ps
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

        proj_geom, sino = create_sinogram(img, 512, 180)

        dart = DART(
            proj_geom=proj_geom,
            sinogram=sino,
            img_shape=img.shape,
            supersampling_a=1,
            reconstruction_iterations=100,
        )

        for p_val in ps:
            rec_img = dart.run(p=p_val, gray_intensities=grey_intensities, iterations=100)
            rnmp = calc_rnmp(img, rec_img)
            ssim = calc_ssim(img, rec_img)

            resuts[p_group][p_val]["rnmp"].append(rnmp)
            resuts[p_group][p_val]["ssim"].append(ssim)
    
        print(f"Finished with phantom {phantom}.")
    print(f"Finished with phantom family {phantom_group}.")


with open("p_dart_results.json", "w") as f:
    json.dump(resuts, f) 
