from source.algorithms import DART, SDART, SSIRT
from source.sinograms.create_sinogram import create_sinogram
from source.utils import saveimg
from source.metrics import calc_rnmp, calc_ssim
import glob
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu



for phantom_group in glob.glob("./phantoms/*"):
    for phantom in glob.glob(f"{phantom_group}/*.png"):
        img = Image.open("./phantoms/blobs/blob_0.png")
        img = np.asarray(img)
    
        proj_geom, sino, = create_sinogram(img, 128, 180, supersampling_a=4)

        sirt_1 = SSIRT.SSIRT(
            proj_geom=proj_geom,
            sinogram=sino,
            img_shape=img.shape,
            supersampling_a=1
        )
        sirt_4 = SSIRT.SSIRT(
            proj_geom=proj_geom,
            sinogram=sino,
            img_shape=img.shape,
            supersampling_a=4
        )
        sirt_8 = SSIRT.SSIRT(
            proj_geom=proj_geom,
            sinogram=sino,
            img_shape=img.shape,
            supersampling_a=8
        )

        # Show off results

        sirt_res = sirt.run(iterations=100)
        saveimg(sirt_res, "sirt_base.png")
        saveimg(dart_res, "dart_base.png")
        
        breakpoint()
        sirt_thresh = threshold_otsu(sirt_res)
        saveimg(sirt_thresh, "sirt_thresh.png")
        
        breakpoint()

