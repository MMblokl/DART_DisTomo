import astra
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from copy import copy
from sinograms.create_sinogram import create_sinogram
from sinograms.sinograms import saveimg

class DART():
    def __init__(
            self,
            proj_geom,
            sinogram,
            img_shape,
            sirt_iterations,
        ):
        self.sirt_iterations = sirt_iterations
        self.img_shape = img_shape

        # Create volume geometry.
        self.proj_geom = proj_geom
        self.vol_geom = astra.create_vol_geom(img_shape)
        self.projector_id = astra.create_projector('cuda', proj_geom, self.vol_geom)

        # Save sinogram into a data2d object
        self.sinogram = sinogram
        self.sino_id = astra.data2d.create('-sino', self.proj_geom, data=sinogram)


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


    def _free_pixels(self):
        # We sample a random number of free pixels according to self.p
        # p - (1 - p) chance to be or not to be included from boundary pixels
        output = np.random.choice([0,1], self.img_shape, p=[self.p, 1-self.p])

        return output


    def gray_thresholds(self, gray_levels):
        """Define threshold array based on input gray_values."""

        # Formula: Tau_i = ( rho_i + rho_i+1 ) / 2
        # Pad with 0 and 255 at start and end
        thresholds = [0] + [
            (gray_levels[i] + gray_levels[i+1]) / 2
            for i in range(len(gray_levels) - 1)
        ] + [255]
        return thresholds


    def _segment(self, inp, thresholds, gray_levels):
        output_img = np.zeros(inp.shape, dtype=np.uint8)

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
        # Create the SIRT config

        config = astra.astra_dict(alg_name)
        config["option"] = {}
        
        # None if this is the first initial reconstruction
        if free_pixels is not None:
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
            reconstruction_id = astra.data2d.create("-vol", self.vol_geom, data=0.0)
            config["ProjectionDataId"] = self.sino_id

        config["ReconstructionDataId"] = reconstruction_id
        config["option"].update({'MinConstraint': 0.0, 'MaxConstraint': 255.0})
        
        # Run the algorithm
        alg_id = astra.algorithm.create(config=config)
        astra.algorithm.run(alg_id, iterations=self.sirt_iterations)

        # Retrieve reconstruction
        reconstruction = astra.data2d.get(reconstruction_id)

        astra.algorithm.delete(alg_id)
        
        return reconstruction
    

    def run(self, p: int, gray_levels: list, iterations: int):
        if p:
            self.p = p

        # Define thresholds for pre-defined gray values
        thresholds = self.gray_thresholds(gray_levels)

        # Initial reconstruction using sirt
        reconstruction = self.reconstruct(
            reconstruction=np.zeros(shape=self.img_shape),
        )

        # Iteration loop
        for i in range(iterations):

            # Segmentation of current reconstruction
            segmentation = self._segment(inp=reconstruction, thresholds=thresholds, gray_levels=gray_levels)

            # Determine border pixels
            border_pixels = self._border_detect(segmentation)

            # Free pixels whatever the fuck that means
            free_pixels = self._free_pixels() | border_pixels

            # Fixed pixels
            fixed_pixels = free_pixels == 0
            # Replace fixed pixels with those from the segmentation; we keep them fixed.
            reconstruction[fixed_pixels] = segmentation[fixed_pixels]
            
            # SIRT RECONSTRUCTION USING FREE PIXELS as a mask
            reconstruction = self.reconstruct(
                reconstruction=reconstruction,
                free_pixels=free_pixels,
            )

            # Smoothing
            if i != iterations:
                smoothed_recon = self._smoothing(reconstruction)
                reconstruction[free_pixels[0], free_pixels[1]] = smoothed_recon[free_pixels[0], free_pixels[1]]
        return reconstruction


if __name__ == "__main__":
    img = Image.open("./blobs/blob_0.png")
    img = np.asarray(img)
    
    proj_geom, sino = create_sinogram(img, 512, 32)

    dart = DART(proj_geom=proj_geom, sinogram=sino, img_shape=img.shape, sirt_iterations=25)
    reconstructed_image = dart.run(0.4, [0,120,255], 100)
    saveimg(reconstructed_image, "./yuh.png")
