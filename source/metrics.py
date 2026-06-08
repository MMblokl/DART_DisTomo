import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def calc_rnmp(reconstruction: np.ndarray, true_img: np.ndarray):
    n_pixels = reconstruction.size
    n_match = (reconstruction != true_img).sum()
    rnmp = n_match / n_pixels

    return rnmp


def calc_ssim(reconstruction: np.ndarray, true_img: np.ndarray):
    val = ssim(im1=true_img, im2=reconstruction, data_range=reconstruction.max() - reconstruction.min())

    return val

if __name__ == "__main__":
    one = np.asarray(Image.open("./blobs/blob_0.png"))
    two = np.asarray(Image.open("./blobs/blob_1.png"))

    print(calc_rnmp(one, two))
    print(calc_ssim(one, two))