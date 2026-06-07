from source.algorithms import DART, SDART, SSIRT
from source.sinograms.create_sinogram import create_sinogram
from source.utils import saveimg
from source.metrics import calc_rnmp, calc_ssim
import glob
from PIL import Image
import numpy as np


a_vals = [0,4,8,16]
base_dict = {"ssirt": [], "dart": [], "sdart": []}

for phantom_group in glob.glob("./phantoms/*"):
    results = {
        "blob": {
            1: {"ssirt": [], "dart": [], "sdart": []},
            4: {"ssirt": [], "dart": [], "sdart": []},
            8: {"ssirt": [], "dart": [], "sdart": []},
            16: {"ssirt": [], "dart": [], "sdart": []}
        },
        "bone": {
            1: {"ssirt": [], "dart": [], "sdart": []},
            4: {"ssirt": [], "dart": [], "sdart": []},
            8: {"ssirt": [], "dart": [], "sdart": []},
            16: {"ssirt": [], "dart": [], "sdart": []}
        },
        "mesh": {
            1: {"ssirt": [], "dart": [], "sdart": []},
            4: {"ssirt": [], "dart": [], "sdart": []},
            8: {"ssirt": [], "dart": [], "sdart": []},
            16: {"ssirt": [], "dart": [], "sdart": []}
        },
    }
    for phantom in glob.glob(f"{phantom_group}/*.png"):
        phantom = "./phantoms/meshes/mesh_0.png"
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
    
        proj_geom, sino, = create_sinogram(img, 64, 180)

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
                lambda_hp=0.24,
                reconstruction_iterations=10,
            )

            ssirt_res = ssirt.run(gray_intensities=grey_intensities, iterations=100)
            dart_res = dart.run(p=0.4, gray_intensities=grey_intensities, iterations=100)
            sdart_res = sdart.run(gray_intensities=grey_intensities, iterations=100)

            results[p_group][a_val]["ssirt"].append({"rnmp": calc_rnmp(img, ssirt_res), "ssim": calc_ssim(img, ssirt_res)})
            results[p_group][a_val]["dart"].append({"rnmp": calc_rnmp(img, dart_res), "ssim": calc_ssim(img, dart_res)})
            results[p_group][a_val]["sdart"].append({"rnmp": calc_rnmp(img, sdart_res), "ssim": calc_ssim(img, sdart_res)})

