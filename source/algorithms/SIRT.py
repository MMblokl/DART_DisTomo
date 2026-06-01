import astra
import numpy as np
from copy import copy
from PIL import Image
from source.sinograms.create_sinogram import create_sinogram
from source.utils import saveimg
from source.metrics import calc_rnmp, calc_ssim

class SIRT:
    def __init__(
            self,
            proj_geom,
            sinogram,
            img_shape,
            supersampling_a = None,
        ):
        self.supersampling_a = supersampling_a
        self.img_shape = img_shape

        # Create volume geometry.
        self.proj_geom = proj_geom
        self.vol_geom = astra.create_vol_geom(img_shape)
        if self.supersampling_a:
            self.projector_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom, options={"DetectorSuperSampling": supersampling_a})
        else:
            self.projector_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        
        # Save sinogram into a data2d object
        self.sinogram = sinogram
        self.sino_id = astra.data2d.create('-sino', self.proj_geom, data=sinogram)
    
    def run(self, iterations: int):
        """ Create a SIRT reconstruction for the given input image, free_pixel mask.
        If free_pixels is None, then an initial reconstruction from the sinogram in self.sino is
        taken with an empty prior reconstruction.
        
        Args:
            reconstruction (np.ndarray | None): Prior reconstructed input image.
            free_pixels (np.ndarray | None): Input mask for each free pixel in the input
        """

        # Create the SIRT config

        config = astra.astra_dict("SIRT_CUDA")
        config["option"] = {}
        
        # None if this is the first initial reconstruction
        # If there is no free_pixel_mask given, we make a blank image from the vol_geom
        reconstruction_id = astra.data2d.create("-vol", self.vol_geom, data=0.0)
        config["ProjectionDataId"] = self.sino_id
        if self.supersampling_a:
            config["option"].update({"DetectorSuperSampling": self.supersampling_a})

        config["ReconstructionDataId"] = reconstruction_id
        config["option"].update({'MinConstraint': 0.0, 'MaxConstraint': 255.0})
        
        # Run the algorithm
        alg_id = astra.algorithm.create(config=config)
        astra.algorithm.run(alg_id, iterations=iterations)

        # Retrieve reconstruction
        reconstruction = astra.data2d.get(reconstruction_id)
        
        # Cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reconstruction_id)

        return reconstruction


if __name__ == "__main__":
    img = Image.open("./phantoms/meshes/mesh_0.png")
    img = np.asarray(img)
    
    proj_geom, sino = create_sinogram(img, 128, 512, supersampling_a=None)

    dart = SIRT(proj_geom=proj_geom, sinogram=sino, img_shape=img.shape, supersampling_a=None)
    reconstructed_image = dart.run(iterations=100)

    print("RNMP, SSIM")
    print(calc_rnmp(img, reconstructed_image), calc_ssim(img, reconstructed_image))

    saveimg(reconstructed_image, "./base.png")
