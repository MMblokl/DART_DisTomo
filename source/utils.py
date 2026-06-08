import numpy as np
import PIL.Image as Image
import astra

def rescale(array):
    """Saves ndarray as PIL image png.

    Args:
        Array (numpyp.ndarray): Input image
    
    Returns:
        Array clipped to (0-255) grayscale values.
    """
    array[array < 0] = 0 # Remove any under zero values
    #array = array/array.max() # Rescale to 0-1
    array = np.clip(array, 0, 255).astype(np.uint8) # Clip to 0,255 range
    
    return array


def saveimg(array: np.ndarray, name: str):
    """Saves ndarray as PIL image png.

    Args:
        Array (np.ndarray): Input image
        Name (string): Location to save
    """
    array = array.astype(np.uint8)
    array = Image.fromarray(array)
    array.save(name)


# Function taken from astra.creators.create_sino, added an option for supersampling for DetectorSuperSampling.
def create_fp(data, proj_id, returnData=True, gpuIndex=None, supersampling_a: int | None = None):
    """Create a forward projection of an image (2D).

    :param data: Image data or ID.
    :type data: :class:`numpy.ndarray` or :class:`int`
    :param proj_id: ID of the projector to use.
    :type proj_id: :class:`int`
    :param returnData: If False, only return the ID of the forward projection.
    :type returnData: :class:`bool`
    :param gpuIndex: Optional GPU index.
    :type gpuIndex: :class:`int`
    :param supersampling_a: Supersampling_a value.
    :type supersampling_a:class`int`|`None`
    :returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`)

    If ``returnData=False``, returns the ID of the forward
    projection. Otherwise, returns a tuple containing the ID of the
    forward projection and the forward projection itself, in that
    order.
"""
    proj_geom = astra.projector.projection_geometry(proj_id)
    vol_geom = astra.projector.volume_geometry(proj_id)

    if isinstance(data, np.ndarray):
        volume_id = astra.data2d.create('-vol', vol_geom, data)
    else:
        volume_id = data
    sino_id = astra.data2d.create('-sino', proj_geom, 0)
    if astra.projector.is_cuda(proj_id):
        algString = 'FP_CUDA'
    else:
        algString = 'FP'
    cfg = astra.astra_dict(algString)
    cfg["option"] = {}
    cfg['ProjectorId'] = proj_id
    if gpuIndex is not None:
        cfg['option'].update({'GPUindex': gpuIndex})
    if supersampling_a is not None:
        cfg["option"].update({"DetectorSuperSampling": supersampling_a})
    cfg['ProjectionDataId'] = sino_id
    cfg['VolumeDataId'] = volume_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    astra.algorithm.delete(alg_id)

    if isinstance(data, np.ndarray):
        astra.data2d.delete(volume_id)
    if returnData:
        return sino_id, astra.data2d.get(sino_id)
    else:
        return sino_id


def create_sinogram(
        img: np.ndarray,
        n_detectors: int,
        n_projections: int,
        supersampling_a: int | None = None
    ):
    """
    Create the sinogram from input image and input parameters.
    The width of each detector element is dependant on the number of detectors and the resolution
    of the phantom. If the phantom image resolution is not the same for both dimension and if the ratio
    between this resolution and the number of detectors isnt a full integer, this function will not work.

    Args:
        img (:class:`numpy.ndarray`): Input image.
        n_detectors (:class:`int`): Number of detector elements.
        n_projections (:class:`int`): Number of projections to sample the image from.
        supersampling_a (:class:`int`): DetectorSuperSampling option in the astra projector.
            Defaults to 1, and decides how many rays are projected per detector.
    Returns:
        Astra projection geometry of the image,
        Sinogram as numpy array.
    """
    # Put image into 2d data object
    vol_geom = astra.create_vol_geom(*img.shape) 
    data_id = astra.data2d.create('-vol', vol_geom, img)
    
    # Makes sure the plane of detectors spans the entire pixel grid resolution / number of detectors.
    spacing = img.shape[0] / n_detectors
    # Supersampling, make each pixel in the image grid have a projection ray.
    if supersampling_a is None:
        supersampling_a = int(spacing)

    # Projections using np.linspace, cutting off the last element of np.pi
    proj_geom = astra.create_proj_geom('parallel', spacing, n_detectors, np.linspace(0, np.pi, n_projections + 1)[:-1])
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom, options={"DetectorSuperSampling": supersampling_a})
    # Create the sinogram
    sino_id, sino = create_fp(data_id, proj_id, supersampling_a=supersampling_a)

    # Clean up of data
    astra.data2d.delete([sino_id, data_id])

    return proj_geom, sino