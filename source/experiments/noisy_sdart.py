from source.metrics import calc_rnmp, calc_ssim
from source.utils import create_sinogram, saveimg
from source.algorithms import SDART
from PIL import Image
import numpy as np
import astra
import json


# Noisy phantom visualization for bone phantoms with ssim
grey_intensities = [0, 110, 150, 220]
phantom = "./phantoms/bones/bone_5.png"
img = Image.open(phantom)
img = np.asarray(img)

# Calculate sinogram and add poisson noise with a set seed.
proj_geom, sino = create_sinogram(img, 64, 180)
sino = astra.functions.add_noise_to_sino(sino, 1e5, seed=202667)

# 100 LSQR iterations
sdart_100 = SDART.SDART(
    proj_geom=proj_geom,
    sinogram=sino,
    img_shape=img.shape,
    supersampling_a=4,
    lambda_hp=0.1,
    reconstruction_iterations=100,
)

# 25 LSQR iterations
sdart_25 = SDART.SDART(
    proj_geom=proj_geom,
    sinogram=sino,
    img_shape=img.shape,
    supersampling_a=4,
    lambda_hp=0.1,
    reconstruction_iterations=25,
)

resdart_100 = sdart_100.run(grey_intensities=grey_intensities, iterations=100)
resdart_25 = sdart_25.run(grey_intensities=grey_intensities, iterations=100)

saveimg(resdart_100, f"./noisy_sdart_100iter4a.png")
saveimg(resdart_25, f"./noisy_sdart_25iter4a.png")

results = {
    25: {"ssim": calc_ssim(img, resdart_25), "rnmp": calc_rnmp(img, resdart_25)},
    100: {"ssim": calc_ssim(img, resdart_100), "rnmp": calc_rnmp(img, resdart_100)}
}

with open("./noisy_sdart_overestimation.json", "w") as f:
    json.dump(results, f) 