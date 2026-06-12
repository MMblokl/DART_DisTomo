import astra
import numpy as np

class SSIRT:
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
    

    def gray_thresholds(
            self,
            gray_intensities: list | tuple,
        ) -> list:
        """Calculated gray-level intensity thresholds for segmentation based on the formula:
        `Tau_i = ( rho_i + rho_i+1 ) / 2`
        Where `Tau` is the threshold for `gray-value` `rho_i` on position `i` in the array.
        
        Args:
            gray_intensities (list | tuple): List of known gray levels in the image from low to high.
        
        Returns:
            List of gray value thresholds according to the formula.
        """
        # 
        # Pad with 0 and 255 at start and end
        thresholds = [0] + [
            (gray_intensities[i] + gray_intensities[i+1]) / 2
            for i in range(len(gray_intensities) - 1)
        ] + [255]

        return thresholds


    def segment(
            self,
            inp: np.ndarray,
            thresholds: list | tuple,
            gray_intensities: list | tuple,
        ) -> np.ndarray:
        """ Creates a simple segmentation according to the thresholds given in thresholds.

        Args:
            inp (numpy.ndarray): Input image.
            threshold (list | tuple): Array of thresholds for each gray level value.
            gray_intensities (list | tuple):  Array of prior gray levels defined by the user for each threshold.
        
        Returns:
            Segmented image where each pixel from in the input image is thresholded for
            each threshold `i` in thresholds, and replaced by `gray_intensities[i]` if it is in
            between the current and next threshold.
        """
        output_img = np.zeros(inp.shape, dtype=np.uint8)

        for i, threshold in enumerate(thresholds[:-1]):
            # At each pixel of the input, get indexes that are both above current threshold and are not above the next threshold
            segmentation = (inp >= threshold) * (inp <= thresholds[i + 1])
            
            # Fill segmented areas into ouput with gray values for that specific threshold
            output_img[segmentation] = gray_intensities[i]
        
        return output_img


    def run(self, gray_intensities: list, iterations: int) -> np.ndarray:
        """ Create a SIRT reconstruction for the given input image.
        
        Args:
            reconstruction (np.ndarray | None): Prior reconstructed input image.
            iterations (integer): Number of iterations to run the algorithm for.
        
        Returns:
            Numpy ndarray of image reconstruction.
        """

        # Create the SIRT config
        config = astra.astra_dict("SIRT_CUDA")
        config["option"] = {}
        
        # Create an empty reconstruction ID to place it into
        reconstruction_id = astra.data2d.create("-vol", self.vol_geom, data=0.0)
        config["ProjectionDataId"] = self.sino_id
        # If supersampling is supplied to the class, turn it on in the alg config.
        if self.supersampling_a:
            config["option"].update({"DetectorSuperSampling": self.supersampling_a})

        # Set the reconstruction ID and the upper and lower bound to 0, 255
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

        # segment using the same setup as in DART
        # Define thresholds for pre-defined gray values
        thresholds = self.gray_thresholds(gray_intensities)
        segmentation = self.segment(inp=reconstruction, thresholds=thresholds, gray_intensities=gray_intensities)

        return segmentation
