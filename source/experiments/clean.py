from source.algorithms import DART, SDART, SIRT
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
    
        proj_geom, sino, proj_id = create_sinogram(img, 128, 32)

        sirt = SIRT.SIRT(
            proj_geom=proj_geom,
            proj_id=proj_id,
            sinogram=sino,
            img_shape=img.shape,
            supersampling_a=4
        )
        dart = DART.DART(
            proj_geom=proj_geom,
            proj_id=proj_id,
            sinogram=sino,
            img_shape=img.shape,
            reconstruction_iterations=10,
            supersampling_a=4
        )

        dart_res = dart.run(iterations=100, gray_intensities=[0, 110, 255], p=0.4)
        sirt_res = sirt.run(iterations=100)
        saveimg(sirt_res, "sirt_base.png")
        saveimg(dart_res, "dart_base.png")
        
        breakpoint()
        sirt_thresh = threshold_otsu(sirt_res)
        saveimg(sirt_thresh, "sirt_thresh.png")
        
        breakpoint()

