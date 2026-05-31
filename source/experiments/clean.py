from source.algorithms import DART, SDART, SIRT
from source.sinograms.create_sinogram import create_sinogram
import glob
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu


for phantom_group in glob.glob("./phantoms/*"):
    for phantom in glob.glob(f"{phantom_group}/*.png"):
        img = Image.open("./phantoms/blobs/blob_0.png")
        img = np.asarray(img)
    
        proj_geom, sino, proj_id = create_sinogram(img, 128, 32)

        sirt_res = SIRT.SIRT(
            proj_geom=proj_geom,
            proj_id=proj_id,
            sinogram=sino,
            img_shape=img.shape,
            reconstruction_iterations=100,
            supersampling_a=4
        )
        
        breakpoint()

