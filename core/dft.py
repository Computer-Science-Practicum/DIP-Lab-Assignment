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
        dft.data = np.zeros(self.data.shape, dtype=complex)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if np.sqrt((i - self.data.shape[0]//2)**2 + (j - self.data.shape[1]//2)**2) > radius:
                    dft.data[i,j] = self.data[i,j]

        return dft
    