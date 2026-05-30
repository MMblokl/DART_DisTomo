import astra
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import lsqr, LinearOperator
from copy import copy
from source.sinograms.create_sinogram import create_sinogram
from source.utils import rescale, saveimg

class SDART():
    def __init__(
            self,
            proj_geom,
            sinogram,
            img_shape,
            reconstruction_iterations,
            lambda_hp,
        ):
        self.reconstruction_iterations = reconstruction_iterations
        self.img_shape = img_shape
        self.lambda_hp = lambda_hp

        # Create volume geometry.
        self.proj_geom = proj_geom
        self.vol_geom = astra.create_vol_geom(img_shape)
        self.projector_id = astra.create_projector('cuda', proj_geom, self.vol_geom)

        # Save sinogram into a data2d object
        self.sinogram = sinogram
        self.sino_id = astra.data2d.create('-sino', self.proj_geom, data=sinogram)


    def calculate_B(self, inp: np.ndarray):
        """Calculation of matrix B which contains all neighbour difference values.
        
        Args:
            inp (np.ndarray): Input image as ndarray.

        Returns:
            int8 matrix for each pixel position
        """        
        # Simple image padding to prevent index errors
        pad = np.pad(inp, 1, mode="edge")

        # This is basically just a semi-vectorized method of checking each pixel.
        # This checks counts all non-equal border pixels and generates a matrix for each position.
        count = (
            (inp != pad[:-2, :-2]).astype(np.uint8) +   # Top left
            (inp != pad[:-2, 1:-1]).astype(np.uint8) + # Top
            (inp != pad[:-2, 2:]).astype(np.uint8) +   # Top right
            (inp != pad[1:-1, :-2]).astype(np.uint8) + # Left
            (inp != pad[1:-1, 2:]).astype(np.uint8) +  # Right
            (inp != pad[2:, :-2]).astype(np.uint8) +   # Bottom left
            (inp != pad[2:, 1:-1]).astype(np.uint8) +  # Bottom
            (inp != pad[2:, 2:]).astype(np.uint8)      # Bottom right
        )
        return count


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


    def lsqr_recon(self, B, W, v):
        """Reconstruction using B, W and v with a least squares problem solver.
        
        Args:
            B (np.ndarray): Matrix of neighbour values from calc_b.
            W: W matrix from astra.opTomo() calculated from the projector space.
            v (np.ndarray): Segmented image to be used in calculating output
        
        Returns:
            reconstructed image based on inputs.
        """
        d = 100 / (3 ** B.ravel())
        m, n = W.shape

        # This is the A*x part
        def matvec(x):
            return np.concatenate(
                [
                    W @ x,
                    self.lambda_hp * (d * x)
                ]
            )
        
        # The A'*b part
        def rmatvec(b):
            b1 = b[:m]
            b2 = b[m:]

            return (
                W.T @ b1 +
                self.lambda_hp * (d * b2)
            )

        # Construct A
        A = LinearOperator(
            shape=(m + n, n),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float32,
        )

        # B component
        right = np.concatenate(
            [
                self.sinogram.flatten(),
                self.lambda_hp * ( d * v.flatten() )
            ]
        )

        # Use lsqr with the same number of iterations as SIRT for DART.
        reconstruction = lsqr(A, right, iter_lim=self.reconstruction_iterations)[0]
        reconstruction = reconstruction.reshape(512,512)
        
        # Rescale to 255
        reconstruction = rescale(reconstruction)

        return reconstruction


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
        ) -> np.ndarray:
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
        
        # Define thresholds for pre-defined gray values
        thresholds = self.gray_thresholds(gray_intensities)

        # Initial reconstruction using sirt
        reconstruction = self.reconstruct()

        # Initial segmentation
        segmentation = self.segment(inp=reconstruction, thresholds=thresholds, gray_intensities=gray_intensities)

        # Iteration loop
        for _ in range(iterations):
            # Segmentation of current reconstruction
            v = segmentation

            # Determine Penalty for each pixel
            B = self.calculate_B(segmentation)
            #D = self.calculate_D(B)

            # Calculate the W matrix
            W = astra.optomo.OpTomo(self.projector_id)

            reconstruction = self.lsqr_recon(B, W, v)

            # Segment again
            segmentation = self.segment(inp=reconstruction, thresholds=thresholds, gray_intensities=gray_intensities)

        # Cleanup
        astra.data2d.delete(self.sino_id)
        astra.projector.delete(self.projector_id)

        return segmentation


if __name__ == "__main__":
    img = Image.open("./phantoms/blobs/blob_0.png")
    img = np.asarray(img)
    
    proj_geom, sino = create_sinogram(img, 512, 32)

    sdart = SDART(proj_geom=proj_geom, sinogram=sino, img_shape=img.shape, reconstruction_iterations=25, lambda_hp=0.24)
    reconstructed_image = sdart.run(0.4, [0,120,255], 100)
    saveimg(reconstructed_image, "./yuh.png")
