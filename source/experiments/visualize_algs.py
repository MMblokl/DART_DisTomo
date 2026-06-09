from PIL import Image
import numpy as np
import os
from source.algorithms import SSIRT, SDART, DART
from source.utils import saveimg, create_sinogram
import astra

if not os.path.exists("./visuals/"):
    os.makedirs("./visuals/")

a_vals = [1,4,8,16]
grey_intensities = [0,255]

# Open image and sample sinogram
phantom = "./phantoms/meshes/mesh_7.png"
img = Image.open(phantom)
img = np.asarray(img)
proj_geom, sino = create_sinogram(img, 64, 180)
saveimg(sino, f"./visuals/sinogram_clean.png")
saveimg(astra.functions.add_noise_to_sino(sino, 1e5, seed=202667), "./visuals/sinogram_noisy.png")


# For each supersampling a-value, save one image per alg.
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
    ssirt_res = ssirt.run(gray_intensities=grey_intensities, iterations=100)
    dart_res = dart.run(p=0.4, gray_intensities=grey_intensities, iterations=100)
    sdart_res = sdart.run(gray_intensities=grey_intensities, iterations=100)

    saveimg(ssirt_res, f"./visuals/ssirt_{a_val}.png")
    saveimg(dart_res, f"./visuals/dart_{a_val}.png")
    saveimg(sdart_res, f"./visuals/sdart_{a_val}.png")