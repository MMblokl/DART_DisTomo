import astra
import numpy as np
from copy import copy

class SIRT:
    def __init__(
            self,
            proj_geom,
            proj_id,
            sinogram,
            img_shape,
            reconstruction_iterations,
            supersampling_a = None,
        ):
        self.reconstruction_iterations = reconstruction_iterations
        self.supersampling_a = supersampling_a
        self.img_shape = img_shape

        # Create volume geometry.
        self.proj_geom = proj_geom
        self.vol_geom = astra.create_vol_geom(img_shape)
        self.projector_id = proj_id

        # Save sinogram into a data2d object
        self.sinogram = sinogram
        self.sino_id = astra.data2d.create('-sino', self.proj_geom, data=sinogram)
    
    def run(self):
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
        astra.algorithm.run(alg_id, iterations=self.reconstruction_iterations)

        # Retrieve reconstruction
        reconstruction = astra.data2d.get(reconstruction_id)
        
        # Cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reconstruction_id)

        return reconstruction