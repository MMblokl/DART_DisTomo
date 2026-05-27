import astra
import numpy as np

def create_sinogram(img, n_detectors, n_projections):
    # Put image into 2d data object
    vol_geom = astra.create_vol_geom(*img.shape) 
    data_id = astra.data2d.create('-vol', vol_geom, img)
        
    # Projections using np.linspace, cutting off the last element of np.pi
    proj_geom = astra.create_proj_geom('parallel', 1.0, n_detectors, np.linspace(0, np.pi, n_projections + 1)[:-1])
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
        
    # Create the sinogram
    sino_id, sino = astra.create_sino(data_id, proj_id)

    # Clean up of data
    astra.data2d.delete([sino_id, data_id])
    astra.projector.delete(proj_id)

    return proj_geom, sino