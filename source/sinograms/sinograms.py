from PIL import Image
import astra
import numpy as np
import glob
from os import mkdir
from os.path import isdir

seed = 167
np.random.seed(seed)

# Create dir for phantoms
if not isdir("./sinograms/"):
    mkdir("./sinograms/")
# Create dir for phantoms
if not isdir("./fbp/"):
    mkdir("./fbp/")
if not isdir("./sirt/"):
    mkdir("./sirt/")


def saveimg(array, name):
    """Saves ndarray as PIL image png.

    Args:
        Array (np.ndarray): Input image
        Name (string): Location to save
    """
    array[array < 0] = 0 # Remove any under zero values
    array = array/array.max() # Rescale to 0-1
    array = np.clip(array*255, 0, 255).astype(np.uint8) # Clip to 0,255 range
    array = Image.fromarray(array)
    array.save(name)


# Parameters:
# N_projections
# Number of detectors
# Number of SIRT cycles
# Boolean of noise yes/no
parameter_list = [ 
    [128, 512, 20, False],
    [512, 512, 100, False],
    [128, 512, 20, True],
    [512, 512, 100, True],
]

# Read all images:
iteration = 1
for n_projections, n_detectors, n_sirt_iterations, noise in parameter_list:
    for path in glob.glob("./phantoms/*.png"):
        img = Image.open(path)
        img = np.asarray(img)
        
        # Put image into 2d data object
        vol_geom = astra.create_vol_geom(*img.shape) 
        data_id = astra.data2d.create('-vol', vol_geom, img)
        
        # Projections using np.linspace, cutting off the last element of np.pi
        proj_geom = astra.create_proj_geom('parallel', 1.0, n_detectors, np.linspace(0, np.pi, n_projections + 1)[:-1])
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
        
        # Create sinogram
        sino_id, sino = astra.create_sino(data_id, proj_id)

        # Add poisson noise and overwrite the sinogram object on the GPU
        if noise:
            sino = astra.functions.add_noise_to_sino(sino, 1e5, seed=seed)
            astra.data2d.store(sino_id, sino)

        # Save sinogram as a png
        saveimg(sino, f"./sinograms/Iteration{iteration}_sino_{path.split('/')[-1]}")

        # Backprojection of 2D sinogram
        # FBP
        recon_id = astra.data2d.create("-vol", vol_geom, data=1.0)
        cfg = astra.astra_dict("FBP_CUDA")
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = recon_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Retrieve as ndarray from data2d object
        recon_fbp = astra.data2d.get(recon_id)
        saveimg(recon_fbp, f"./fbp/Iteration{iteration}_fbp_{path.split('/')[-1]}")

        # SIRT reconstruction
        recon_id = astra.data2d.create("-vol", vol_geom, data=1.0)
        cfg = astra.astra_dict("SIRT_CUDA")
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = recon_id
        cfg["option"] = {'MinConstraint': 0.0}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations=n_sirt_iterations)

        # Retrieve as arr and save
        recon_fbp = astra.data2d.get(recon_id)
        saveimg(recon_fbp, f"./sirt/Iteration{iteration}_sirt_{path.split('/')[-1]}")

        # Cleanup of data
        astra.data2d.delete([sino_id, recon_id, data_id])
        astra.projector.delete(proj_id)
        astra.algorithm.delete(alg_id)

    iteration += 1