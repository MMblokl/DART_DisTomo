import astra
import numpy as np


# Function copied from astra toolbox with an option for supersampling.
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


def create_sinogram(img, n_detectors, n_projections, supersampling_a: int | None = None):
    # Put image into 2d data object
    vol_geom = astra.create_vol_geom(*img.shape) 
    data_id = astra.data2d.create('-vol', vol_geom, img)
    
    # Makes sure the plane of detectors spans the entire pixel grid resolution / number of detectors.
    spacing = img.shape[0] / n_detectors
    # Supersampling, make each pixel in the image grid have a projection ray.
    if supersampling_a is None:
        supersampling_a = spacing

    # Projections using np.linspace, cutting off the last element of np.pi
    proj_geom = astra.create_proj_geom('parallel', spacing, n_detectors, np.linspace(0, np.pi, n_projections + 1)[:-1])
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom, options={"DetectorSuperSampling": supersampling_a})
    # Create the sinogram
    sino_id, sino = create_fp(data_id, proj_id, supersampling_a=supersampling_a)

    # Clean up of data
    astra.data2d.delete([sino_id, data_id])

    return proj_geom, sino