from source.algorithms.DART import DART
from source.utils import create_sinogram
from source.metrics import calc_rnmp, calc_ssim
from PIL import Image
import numpy as np
import glob
import json

ps = [0.1, 0.2, 0.4, 0.8]
phantoms = ["blob", "bone", "mesh"]

results = {
    phantom: { p_val: {"rnmp": [], "ssim": []} for p_val in ps
    } for phantom in phantoms
}
final_results = {
    phantom: { p_val: {"rnmp": 0, "ssim": 0} for p_val in ps
    } for phantom in phantoms
}

for phantom_group in glob.glob("./phantoms/*"):
    for phantom in glob.glob(f"{phantom_group}/*.png"):
        # Selection of phantom grouping based on their intensity values.
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
            # Sample sinogram
            proj_geom, sino = create_sinogram(img, 64, 180)

            dart = DART(
                proj_geom=proj_geom,
                sinogram=sino,
                img_shape=img.shape,
                supersampling_a=1,
                reconstruction_iterations=100,
            )
            
            # Reconstruct phatnom from sinogram
            rec_img = dart.run(p=p_val, grey_intensities=grey_intensities, iterations=100)
            
            # Save metrics to dict
            rnmp = calc_rnmp(img, rec_img)
            ssim = calc_ssim(img, rec_img)

            results[p_group][p_val]["rnmp"].append(rnmp)
            results[p_group][p_val]["ssim"].append(ssim)
    
        print(f"Finished with phantom {phantom}.")
    print(f"Finished with phantom family {phantom_group}.")

# Average the values over each phantom type/group

for phantom_group in results.keys():
    for p_val in results[phantom_group].keys():
        final_results[phantom_group][p_val]["rnmp"] = np.mean(results[phantom_group][p_val]["rnmp"])
        final_results[phantom_group][p_val]["ssim"] = np.mean(results[phantom_group][p_val]["ssim"])

# Save averaged values to JSON
with open("p_dart_results.json", "w") as f:
    json.dump(final_results, f)
