import astra
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from copy import copy
from source.sinograms.create_sinogram import create_sinogram
from source.utils import saveimg

class DART():
    def __init__(
            self,
            proj_geom,
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
        self.projector_id = astra.create_projector('cuda', proj_geom, self.vol_geom)

        # Save sinogram into a data2d object
        self.sinogram = sinogram
        self.sino_id = astra.data2d.create('-sino', self.proj_geom, data=sinogram)


    def border_detect(self, inp: np.ndarray):
        """Border pixel returns, basically just an edge detection
        
        Args:
            inp (np.ndarray): Input image as ndarray.

        Returns:
            Boolean/binary image with True on edges/borders.
        """        
        # Simple image padding to prevent index errors
        pad = np.pad(inp, 1, mode="edge")

        # This is basically just a semi-vectorized method of checking each pixel.
        # This checks if any of the neighbours of one pixel is different, that pixel is an edge.
        edges = (
            (inp != pad[:-2, :-2]) | # Top left
            (inp != pad[:-2, 1:-1]) | # Top pixel
            (inp != pad[:-2, 2:]) | # Top right
            (inp != pad[1:-1, :-2]) | # Left pixel
            (inp != pad[1:-1, 2:]) | # Right pixel
            (inp != pad[2:, :-2]) | # Bottom left
            (inp != pad[2:, 1:-1]) | # Bottom pixel
            (inp != pad[2:, 2:]) # Bottom right
        )

        return edges


    def smoothing(self, inp: np.ndarray):
        """Smoothing function for after a single reconstruction
        
        Args:
            inp (np.ndarray): Input image
        
        Returns:
            Smoothed input image.
        """
        # Gaussian also works
        output = gaussian_filter(inp, sigma=1)

        return output


    def free_pixels(self):
        """Samples random locations based on image shape where free_pixels will be located.
        
        Args:
            None
        
        Returns:
            np.ndarray of shape self.img_shape with a number of True according 1-self.p, with the rest False.
        """
        # We sample a random number of free pixels according to self.p
        # p - (1 - p) chance to be or not to be included from boundary pixels
        output = np.random.choice([False,True], self.img_shape, p=[self.p, 1-self.p])

        return output


    def gray_thresholds(
            self,
            gray_intensities: list | tuple,
        ) -> list:
        """Define threshold array based on input gray_values.
        Uses formula:
        Tau_i = ( rho_i + rho_i+1 ) / 2
        Where Tau is the threshold for gray value rho on position i in the array.
        
        Args:
            gray_intensities (list | tuple or array_like): List of known gray levels in the image from low to high.
        
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
        ):
        """ Creates a simple segmentation according to the thresholds given in thresholds.

        Args:
            inp (np.ndarray): Input image.
            threshold (list | tuple): Array of thresholds for each gray level value.
            gray_intensities (list | tuple):  Array of prior gray levels defined by the user for each threshold.
        
        Returns:
            Segmented image where each pixel from in the input image is thresholded for
            each threshold i in thresholds, and replaced by gray_intensities[i]if it is in
            between the current and next threshold.
        """
        output_img = np.zeros(inp.shape, dtype=np.uint8)

        for i, threshold in enumerate(thresholds[:-1]):
            # At each pixel of the input, get indexes that are both above current threshold and are not above the next threshold
            segmentation = (inp >= threshold) * (inp <= thresholds[i + 1])
            
            # Fill segmented areas into ouput with gray values for that specific threshold
            output_img[segmentation] = gray_intensities[i]
        
        return output_img


    def reconstruct(
            self,
            reconstruction: np.ndarray | None = None,
            free_pixels: np.ndarray | None = None,
        ):
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
        if free_pixels is not None:
            # Create free_pixels dataid
            free_pix_idx = np.where(free_pixels != 0)
            rec = copy(reconstruction)
            
            # Remove the free pixels from the sinogram
            rec[free_pix_idx] = 0
            
            # Create sino from free pixels
            _, fixed_sinogram = astra.create_sino(rec, self.projector_id)
            
            # Free pixel sinogram is the difference between fixed and main sinogram
            free_pixel_sinogram = self.sinogram - fixed_sinogram

            # Create a data2d object for use in the algorithm
            free_pixel_sinogram_id = astra.data2d.create("-sino", self.proj_geom, data=free_pixel_sinogram)
            reconstruction_id = astra.data2d.create("-vol", self.vol_geom, data=rec)

            # Put the free pixel sinogram into config
            config["ProjectionDataId"] = free_pixel_sinogram_id

            # Put the free pixels as a reconstructionmask into the algorithm
            free_pixels_id = astra.data2d.create("-vol", self.vol_geom, data=free_pixels)
            config["option"] = {"ReconstructionMaskId": free_pixels_id}
        else:
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
        if free_pixels is not None:
            astra.data2d.delete(free_pixel_sinogram_id)
            astra.data2d.delete(free_pixels_id)

        return reconstruction
    

    def run(self, p: int, gray_intensities: list, iterations: int):
        """Run the DART algorithm according to the initializated values.
        
        Args:
            p (integer): Probability of sampling a pixel as a fixed pixel.
            gray_intensities (list | tuple): List of prior gray intensity levels.
            iterations (integer): Number of DART iterations.
        
        Returns:
            Output image after DART reconstruction with set settings.
        """
        
        self.p = p
        # Define thresholds for pre-defined gray values
        thresholds = self.gray_thresholds(gray_intensities)

        # Initial reconstruction using sirt
        reconstruction = self.reconstruct()

        # Segmentation of current reconstruction
        segmentation = self.segment(inp=reconstruction, thresholds=thresholds, gray_intensities=gray_intensities)

        # Iteration loop
        for i in range(iterations):

            # Determine border pixels
            border_pixels = self.border_detect(segmentation)

            # Free pixels whatever the fuck that means
            free_pixels = self.free_pixels() | border_pixels

            # Fixed pixels
            fixed_pixels = free_pixels == 0
            # Replace fixed pixels with those from the segmentation; we keep them fixed.
            reconstruction[fixed_pixels] = segmentation[fixed_pixels]
            
            # SIRT RECONSTRUCTION USING FREE PIXELS as a mask
            reconstruction = self.reconstruct(
                reconstruction=reconstruction,
                free_pixels=free_pixels,
            )

            # Segmentation of current reconstruction
            segmentation = self.segment(inp=reconstruction, thresholds=thresholds, gray_intensities=gray_intensities)

            # Smoothing
            if i != iterations:
                smoothed_recon = self.smoothing(reconstruction)
                reconstruction[free_pixels] = smoothed_recon[free_pixels]
        
        # Cleanup
        astra.data2d.delete(self.sino_id)
        astra.projector.delete(self.projector_id)
        
        return segmentation


if __name__ == "__main__":
    img = Image.open("./phantoms/blobs/blob_0.png")
    img = np.asarray(img)
    
    proj_geom, sino = create_sinogram(img, 128, 32)

    dart = DART(proj_geom=proj_geom, sinogram=sino, img_shape=img.shape, reconstruction_iterations=25)
    reconstructed_image = dart.run(0.4, [0,120,255], 100)
    saveimg(reconstructed_image, "./base.png")
