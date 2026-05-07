import astra
import numpy as np
import PIL
from scipy.ndimage import gaussian_filter
from copy import copy


class DART():
    def __init__(
            self,
            p,
            proj_geom,
            sinogram,
            img_shape,
            projector_id,
            iterations,
        ):
        self.p = p
        self.iterations = iterations
        self.img_shape = img_shape

        
        # Save sinogram into a data2d object
        self.sinogram = sinogram
        self.sino_id = astra.data2d.create('-vol', self.vol_geom, sinogram)

        # Create volume geometry.
        self.proj_geom = proj_geom
        self.vol_geom = astra.create_vol_geom(img_shape)
        self.projector_id = projector_id


    def _border_detect(self, inp: np.ndarray):
        """Border pixel returns, basically just an edge detection"""        
        # Simple image padding to prevent index errors
        pad = np.pad(inp, 1, mode="edge")

        # This is basically just a vectorized method of checking each pixel.
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


    def _smoothing(self, inp):
        # Gaussian also works
        output = gaussian_filter(inp, sigma=1)
        
        # kernel = [
        #     [1, 1, 1],
        #     [1, (1-self.b), 1],
        #     [1,1,1]
        # ] * self.b / 8

        return output


    def _free_pixels(self, inp):
        # We sample a random number of free pixels according to self.p
        # p - (1 - p) chance to be or not to be included from boundary pixels
        output = np.random.choice([0,1], inp.shape, p=[self.p, 1-self.p])


    def gray_thresholds(self, gray_levels):
        """Define threshold array based on input gray_values."""

        # Formula: Tau_i = ( rho_i + rho_i+1 ) / 2
        # Pad with 0 and 255 at start and end
        thresholds = [0] + [
            (gray_levels[i] + gray_levels[i+1]) / 2
            for i in range(gray_levels)
        ] + [255]

        return thresholds


    def _segment(self, inp, thresholds, gray_levels):
        output_img = np.full(inp.shape, dtype=np.uint8)

        for i, threshold in enumerate(thresholds[:-1]):
            # At each pixel of the input, get indexes that are both above current threshold and are not above the next threshold
            segmentation = (inp >= threshold) * (inp <= thresholds[i + 1])
            
            # Fill segmented areas into ouput with gray values for that specific threshold
            output_img[segmentation] = gray_levels[i]
        
        return output_img


    def reconstruct(
            self,
            reconstruction,
            free_pixels=None,
            alg_name="SIRT_CUDA"
        ):
        config = astra.astra_dict(alg_name)
        
        # None if this is the first initial reconstruction
        if free_pixels:
            # Create free_pixels dataid
            free_pix_idx = np.where(free_pixels != 0)
            rec = copy(reconstruction)
            
            # Remove the free pixels from the sinogram
            rec[free_pix_idx[0], free_pix_idx[1]] = 0
            
            # Create sino from free pixels
            _, fixed_sinogram = astra.create_sino(rec, self.projector_id)
            free_sinogram = self.sinogram - fixed_sinogram
            free_pixel_sinogram_id = astra.data2d.create("-sino", self.proj_geom, data=free_sinogram)
            reconstruction_id = astra.data2d.create("-vol", self.vol_geom, data=rec)

            # Put the free pixel sinogram into config
            config["ProjectionDataId"] = free_pixel_sinogram_id

            # Put the free pixels as a reconstructionmask into the algorithm
            free_pixels_id = astra.data2d.create("-vol", self.vol_geom, data=free_pixels)
            config["option"] = {"ReconstructionMaskId": free_pixels_id}
        else:
            reconstruction_id = astra.data2d.create("-vol", self.vol_geom, data=float(0))
            config["ProjectionDataId"] = self.sino_id            

        config["ReconstructionDataId"] = reconstruction_id
        config["option"].update({'MinConstraint': 0.0, 'MaxConstraint': 255.0})
        
        # Run the algorithm
        alg_id = astra.algorithm.create(config=config)
        astra.algorithm.run(alg_id, iterations=self.iterations)

        # Retrieve reconstruction
        reconstruction = astra.data2d.get(reconstruction_id)

        astra.algorithm.delete(alg_id)
        
        return reconstruction
    

    def __call__(self, *args, **kwds):
        if args["p"]:
            self.p = args["p"]
        
        gray_levels = args["gray_levels"]

        # Define thresholds for pre-defined gray values
        thresholds = self.gray_thresholds(gray_levels)

        # Initial reconstruction using sirt
        reconstruction = self.reconstruct(
            reconstruction=np.zeros(shape=self.img_shape),
        )

        # Iteration loop
        for i in range(args["iterations"]):

            # Segmentation of current reconstruction
            segmentation = self._segment(inp=reconstruction, thresholds=thresholds, gray_levels=gray_levels)

            # Determine border pixels
            border_pixels = self._border_detect(segmentation)

            # Free pixels whatever the fuck that means
            free_pixels = self._free_pixels(border_pixels)

            # Fixed pixels
            fixed_pixels = np.where(free_pixels == 0)
            # Replace fixed pixels with those from the segmentation; we keep them fixed.
            reconstruction[fixed_pixels[0], fixed_pixels[1]] = segmentation[fixed_pixels[0], fixed_pixels[1]]
            
            # SIRT RECONSTRUCTION USING FREE PIXELS as a mask
            reconstruction = self.reconstruct(
                reconstruction=reconstruction,
                free_pixels=free_pixels,
            )

            # Smoothing
            if i != args["iterations"]:
                smoothed_recon = self._smoothing(reconstruction)
                reconstruction[free_pixels[0], free_pixels[1]] = smoothed_recon[free_pixels[0], free_pixels[1]]


if __name__ == "__main__":
    dart = DART()
    image = [[0,0,0], [0,1,0], [0,0,0]]
    image = np.asarray(image)
    image = np.pad(image, 1, mode="edge")
    dart._border_detect(inp=image)