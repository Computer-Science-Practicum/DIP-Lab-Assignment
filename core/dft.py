import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import rotate
from core.image import Image

class Matrix:
    def __init__(self) -> None:
        self.data = None

    def rotate(self, angle: float) -> 'Matrix':
        rotated = type(self)()
        rotated.data = rotate(self.data, angle,reshape=True)
        return rotated
    
    def dot(self, matrix: 'Matrix') -> 'Matrix':
        result = type(self)()
        result.data = np.dot(self.data, matrix.data)
        return result
    

class DFT(Matrix):
    def __init__(self,image: Image = None) -> None:
        super().__init__()
        if(image is not None):
            self.compute(image)

    def compute(self, image: Image) -> None:
        self.data = np.fft.fft2(image.data)
        self.data = np.fft.fftshift(self.data)
    
    def inverse(self) -> Image:
        img = Image()
        img.data = np.fft.ifftshift(self.data)
        img.data = np.fft.ifft2(self.data)
        img.data = np.abs(img.data)
        return img
    
    def amplitude(self,log=False) -> Image:
        img = Image()

        img.data = np.abs(self.data)
        if log:
            img.data = np.log(img.data + 1)
        return img
    
    def phase(self) -> Image:
        img = Image()
        img.data = np.angle(self.data)
        return img
    
    def low_pass_filter(self, radius: int) -> 'DFT':
        dft = DFT()
        dft.data = np.zeros(self.data.shape, dtype=complex)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if np.sqrt((i - self.data.shape[0]//2)**2 + (j - self.data.shape[1]//2)**2) < radius:
                    dft.data[i,j] = self.data[i,j]
        return dft
    
    def high_pass_filter(self, radius: int) -> 'DFT':
        dft = DFT()
        dft.data = self.data * self.HIGH_PASS_FILTER(self.data.shape[0], self.data.shape[1], radius)
        dft.data = 0 + dft.data # to remove negative zeros
        return dft
    
    
    @staticmethod
    def HIGH_PASS_FILTER(h, w, radius) -> np.ndarray:
        mask = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                if np.sqrt((i - h//2)**2 + (j - w//2)**2) > radius:
                    mask[i,j] = 1
        return mask
    
    @staticmethod
    def LOW_PASS_FILTER(h, w, radius) -> np.ndarray:
        return 1 - DFT.HIGH_PASS_FILTER(h, w, radius)
    
class Filtering:
    """
    Filtering class contains methods for image filtering in the frequency domain.
    """

    @staticmethod
    def unsharp_masking(image: Image, alpha: float, radius: int, verbose = False) -> Image:

        dft = DFT(image)
        if verbose:
            print("Original Image")
            image.display()
            print("DFT of Original Image")
            dft.amplitude(True).display()
            dft.phase().display()

        dft_filtered = DFT()
        # mask = Image()
        # mask.data = image.data - dft.low_pass_filter(radius).inverse().data
        # if verbose:
        #     print("Mask")
        #     # mask.amplitude(True).display()
        #     # mask.phase().display()
        #     dft.low_pass_filter(radius).inverse().display()
        #     mask.display()

        # dft_filtered.data = dft.data.inverse() + alpha * mask.data

        #entire computation in frequency domain
        dft_filtered.data = (1 +  alpha * DFT.HIGH_PASS_FILTER(image.data.shape[0], image.data.shape[1], radius)) * dft.data

        if verbose:
            print("Filtered DFT")
            dft_filtered.amplitude(True).display()
            dft_filtered.phase().display()
            print("Filtered Image")
            dft_filtered.inverse().display()

        filtered_image = dft_filtered.inverse()
        filtered_image.data = np.clip(filtered_image.data, 0, 1)
        return filtered_image
    
    @staticmethod
    def NOTCH_REJECT_MASK(h, w, x, y, radius) -> np.ndarray:
        mask = np.ones((h,w))
        mask[x-radius:x+radius, y-radius:y+radius] = 0

        mask[x-radius:x+radius, w-y-radius:w-y+radius] = 0
        return mask
    
    @staticmethod
    def NOTCH_PASS_MASK(h, w, x, y, radius) -> np.ndarray:
        return 1 - Filtering.NOTCH_REJECT_MASK(h, w, x, y, radius)

    @staticmethod
    def notch_filter(image: Image, x: int, y: int, radius: int, reject: bool = True, verbose = False) -> Image:
        dft = DFT(image)
        if verbose:
            print("Original Image")
            image.display()
            print("DFT of Original Image")
            dft.amplitude(True).display()
            dft.phase().display()

        dft_filtered = DFT()
        if reject:
            dft_filtered.data = dft.data * Filtering.NOTCH_REJECT_MASK(image.data.shape[0], image.data.shape[1], x, y, radius)
        else:
            dft_filtered.data = dft.data * Filtering.NOTCH_PASS_MASK(image.data.shape[0], image.data.shape[1], x, y, radius)

        if verbose:
            print("Filtered DFT")
            dft_filtered.amplitude(True).display()
            dft_filtered.phase().display()
            print("Filtered Image")
            dft_filtered.inverse().display()

        filtered_image = dft_filtered.inverse()
        filtered_image.data = np.clip(filtered_image.data, 0, 1)
        return filtered_image

    @staticmethod
    def ideal_bandpass_filter(image: Image, low: int, high: int, verbose = False) -> Image:
        dft = DFT(image)
        dft_filtered = dft.high_pass_filter(low)
        dft_filtered = dft_filtered.low_pass_filter(high)

        if verbose:
            print("Original Image")
            image.display()
            print("DFT of Original Image")
            dft.amplitude(True).display()
            dft.phase().display()
            print("Filtered DFT")
            dft_filtered.amplitude(True).display()
            dft_filtered.phase().display()
            print("Filtered Image")
            dft_filtered.inverse().display()

        filtered_image = dft_filtered.inverse()
        filtered_image.data = np.clip(filtered_image.data, 0, 1)
        return filtered_image
    
