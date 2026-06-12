import astra
import numpy as np
from scipy.sparse.linalg import lsqr, LinearOperator
from source.utils import rescale

class SDART():
    def __init__(
            self,
            proj_geom,
            sinogram,
            img_shape,
            reconstruction_iterations,
            lambda_hp,
            supersampling_a=None,
        ):
        self.reconstruction_iterations = reconstruction_iterations
        self.img_shape = img_shape
        self.lambda_hp = lambda_hp
        self.supersampling_a = supersampling_a

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


    def calculate_B(self, inp: np.ndarray):
        """Calculation of matrix B which contains all neighbour difference values.
        
        Args:
            inp (numpy.ndarray): Input image as ndarray.

        Returns:
            Numpy ndarray of the same shape as the input image with neighbour pixel counts at each pixel location.
        """        
        # Simple image padding to prevent index errors
        pad = np.pad(inp, 1, mode="edge")

        # This is basically just a semi-vectorized method of checking each pixel.

        # This is the SAME function as the one in DART, but instead of it being booleans, it converts the booleans of
        # each of the neighbour checks to integers and then sums the resulting checked matrices into a single matrix.
        # The resulting matrix has neighbour counts at the same positition as the ones in "inp".

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


    def lsqr_recon(self, B, W, v):
        """Reconstruction using B, W and v with a least squares problem solver.
        
        Args:
            B (np.ndarray): Matrix of neighbour values from calc_B.
            W: W projection simulation matrix astra.OpTomo() class using a parrallel projector.
            v (np.ndarray): Segmented image to be used in calculating output
        
        Returns:
            reconstructed image based on inputs.
        """
        # Calculation of the 'D' matrix, kept as a single array to reduce memory impact over having it as a sparse matrix.
        d = 100 / (3 ** B.ravel())
        m, n = W.shape

        # This is A*x
        def matvec(x):
            return np.concatenate(
                [
                    W @ x,
                    self.lambda_hp * (d * x)
                ]
            )
        
        # The transposed A'*x
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

        # B component named "right" as B is already taken by the neighbour count matrix.
        right = np.concatenate(
            [
                self.sinogram.flatten(),
                self.lambda_hp * ( d * v.flatten() )
            ]
        )

        # Use lsqr with a fixed number of iterations, the solution of the lsqr is our reconstructed image.
        reconstruction = lsqr(A, right, iter_lim=self.reconstruction_iterations)[0]
        # Solution is a vector, reshape to np.ndarray of correct shape.
        reconstruction = reconstruction.reshape(self.img_shape)
        
        # Rescale to 255 using np.clip
        reconstruction = rescale(reconstruction)

        return reconstruction


    def grey_thresholds(
            self,
            grey_intensities: list | tuple,
        ) -> list:
        """Calculated grey-level intensity thresholds for segmentation based on the formula:
        `Tau_i = ( rho_i + rho_i+1 ) / 2`
        Where `Tau` is the threshold for `grey-value` `rho_i` on position `i` in the array.
        
        Args:
            grey_intensities (list | tuple): List of known grey levels in the image from low to high.
        
        Returns:
            List of grey value thresholds according to the formula.
        """
        # Add 0 and 255 intensities to use as the lower and upper bound
        thresholds = [0] + [
            (
                grey_intensities[i] + grey_intensities[i+1]
            ) / 2
            for i in range(len(grey_intensities) - 1)
        ] + [255]

        return thresholds


    def segment(
            self,
            inp: np.ndarray,
            thresholds: list | tuple,
            grey_intensities: list | tuple,
        ) -> np.ndarray:
        """ Creates a simple segmentation according to the thresholds given in thresholds.

        Args:
            inp (numpy.ndarray): Input image.
            threshold (list | tuple): Array of thresholds for each grey level value.
            grey_intensities (list | tuple):  Array of prior grey levels defined by the user for each threshold.
        
        Returns:
            Segmented image where each pixel from in the input image is thresholded for
            each threshold `i` in thresholds, and replaced by `grey_intensities[i]` if it is in
            between the current and next threshold.
        """
        output_img = np.zeros(inp.shape, dtype=np.uint8)

        for i, threshold in enumerate(thresholds[:-1]):
            # At each pixel of the input, get indexes that are both above current threshold and are not above the next threshold
            segmentation = (inp >= threshold) * (inp <= thresholds[i + 1])
            
            # Fill segmented areas into ouput with grey values for that specific threshold
            output_img[segmentation] = grey_intensities[i]
        
        return output_img


    def reconstruct(self):
        """ Create a SIRT reconstruction as the initial reconstruction to kickstart SDART.

        Returns:
            numpy.ndarray of initial reconstruction.
        """

        # Create the SIRT config
        config = astra.astra_dict("SIRT_CUDA")
        config["option"] = {}

        # Create ID to save reconstruction in
        reconstruction_id = astra.data2d.create("-vol", self.vol_geom, data=0.0)
        config["ProjectionDataId"] = self.sino_id
        config["ReconstructionDataId"] = reconstruction_id

        # Add supersampling to config if active.
        if self.supersampling_a:
            config["option"].update({"DetectorSuperSampling": self.supersampling_a})
        
        # Upper and lower bound of pixel intensities.
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
    

    def run(self, grey_intensities: list, iterations: int) -> np.ndarray:
        """Runs the SDART algorithm according to the initializated values.
        
        Args:
            grey_intensities (list | tuple): List of prior grey intensity levels.
            iterations (integer): Number of SDART iterations.
        
        Returns:
            Output image after SDART reconstruction with set settings.
        """
        # Define thresholds for pre-defined grey values
        thresholds = self.grey_thresholds(grey_intensities)

        # Initial reconstruction using sirt
        reconstruction = self.reconstruct()

        # Initial segmentation
        segmentation = self.segment(inp=reconstruction, thresholds=thresholds, grey_intensities=grey_intensities)

        # Iteration loop
        for _ in range(iterations):
            # Segmentation of current reconstruction
            v = segmentation

            # Determine number of neighbours for each pixel
            B = self.calculate_B(segmentation)

            # Calculate the W matrix
            W = astra.optomo.OpTomo(self.projector_id)

            # Compute the reconstruction using LSQR, D matrix is computed here
            reconstruction = self.lsqr_recon(B, W, v)

            # Segment the result
            segmentation = self.segment(inp=reconstruction, thresholds=thresholds, grey_intensities=grey_intensities)

        # Cleanup
        astra.data2d.delete(self.sino_id)
        astra.projector.delete(self.projector_id)

        return segmentation
