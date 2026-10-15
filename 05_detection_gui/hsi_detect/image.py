from spectral import *
from hsi_detect import utils
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

DATA_EXTS = ('.raw', '.img', '.dat', '.bin', '.bil', '.bip', '.bsq', '')


def resolve_envi_pair(path):
    """
    Return (header_path, data_path) for an ENVI cube, given either the .hdr
    or the binary data file (e.g. .raw). Handles 'scan.hdr' + 'scan.raw' as
    well as 'scan.raw.hdr' + 'scan.raw'.
    """
    path = os.path.abspath(path)
    stem, ext = os.path.splitext(path)
    if ext.lower() == '.hdr':
        hdr = path
        for cand in [stem + e for e in DATA_EXTS] + [stem + e.upper() for e in DATA_EXTS if e]:
            if cand != hdr and os.path.isfile(cand):
                return hdr, cand
        # fall back to a lone .raw file in the same folder
        raws = glob.glob(os.path.join(os.path.dirname(path), '*.raw'))
        if len(raws) == 1:
            return hdr, raws[0]
        raise FileNotFoundError(f"No data file (.raw/.img/.dat...) found for header:\n{hdr}")
    # a data file was picked -> find its header
    data = path
    for cand in (stem + '.hdr', stem + '.HDR', path + '.hdr', path + '.HDR'):
        if os.path.isfile(cand):
            return cand, data
    raise FileNotFoundError(f"No .hdr header found next to data file:\n{data}")


class HyperspectralImage:
    def __init__(self, image_path, smoothing_window=11):
        hdr, data = resolve_envi_pair(image_path)
        self.image_path = hdr
        self.lib = envi.open(hdr, data)
        self.image = self.lib.load()
        self.image = utils.smooth_img_spectrum(self.image, window_size=11)
        self.centers = self.lib.bands.centers
        self.rgb = None

    def make_rgb(self, coefficients=(1,1,1)):
        """
        Creates a RGB image from the HSI image
        
        coefficients (3-tuple): multiples for the Red, Green, and Blue channel,
            respectively. Can be used to tune the color balance in the image.
        """
        self.rgb = utils.bandpass_rgb_function (self.image, 
                                      self.centers, 
                                      coeffs=coefficients)
        # add transparency layer
        alpha_layer = ((1-(np.sum(self.image, axis=2)==0))*255).astype(int)
        alpha_layer = alpha_layer[:,:,np.newaxis]
        self.rgb  = np.concatenate([self.rgb , alpha_layer], axis=2).astype(int)
    
    def flatten(self, normalize=True):
        """
        Return 2D-matrix of image (n_pixels,n_wavelengths)
        """
        image = self.image
        if normalize:
            image = image / np.nanmax(image, axis=2, keepdims=True)
        return np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
    
    def show(self, image=None, savepath=None, **fig_params):
        """
        Visualize RGB image of HSI image.
        If RGB image was not prevously created, it is created.
        """
        if self.rgb is None:
            self.make_rgb()
        
        plt.figure(**fig_params)
        if image is None:
            plt.imshow(self.rgb)
        else:
            plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        if savepath is not None:
            plt.savefig(savepath, transparent=True)
        
        plt.show()