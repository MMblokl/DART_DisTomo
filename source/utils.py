import numpy as np
import PIL.Image as Image

def rescale(array):
    """Saves ndarray as PIL image png.

    Args:
        Array (np.ndarray): Input image
    """
    array[array < 0] = 0 # Remove any under zero values
    array = array/array.max() # Rescale to 0-1
    array = np.clip(array*255, 0, 255).astype(np.uint8) # Clip to 0,255 range
    
    return array


def saveimg(array, name):
    """Saves ndarray as PIL image png.

    Args:
        Array (np.ndarray): Input image
        Name (string): Location to save
    """
    array = Image.fromarray(array)
    array.save(name)