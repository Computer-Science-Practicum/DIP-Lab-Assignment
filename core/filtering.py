from core.image import Image
import numpy as np

class ImageFilters:
    def __init__(self) -> None:
        self.gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    @staticmethod
    def convolve(image: Image, kernel: np.ndarray) -> Image:
        if len(image.data.shape) == 3:
            return ImageFilters.convolve_rgb(image, kernel)

        img = Image()
        img.data = np.zeros(image.data.shape)
        for i in range(1, image.data.shape[0] - 1):
            for j in range(1, image.data.shape[1] - 1):
                img.data[i, j] = np.sum(image.data[i-1:i+2, j-1:j+2] * kernel)
        return img
    
    @staticmethod
    def convolve_rgb(image: Image, kernel: np.ndarray) -> Image:
        img = Image()
        img.data = np.zeros(image.data.shape)

        # Get dimensions
        h, w, c = image.data.shape

        # Iterate over each channel
        for k in range(c):
            # Iterate over each pixel, ignoring borders
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    # Extract the region of interest
                    region = image.data[i-1:i+2, j-1:j+2, k]
                    # Apply the kernel
                    img.data[i, j, k] = np.sum(region * kernel)

        # normalize the image to be in the range [0, 255]
        img.data = (img.data - np.min(img.data)) / (np.max(img.data) - np.min(img.data))


        return img
    
    
    @staticmethod
    def gaussian_blur(image: Image) -> np.ndarray:
        return ImageFilters.convolve(image, ImageFilters().gaussian)
    
    @staticmethod
    def edge_detection(image: Image) -> tuple:
        img_x = ImageFilters.convolve(image, ImageFilters().sobel_x)
        img_y = ImageFilters.convolve(image, ImageFilters().sobel_y)

        #square root of the sum of the squares of the two gradients
        img = Image()
        img.data = np.sqrt(img_x.data**2 + img_y.data**2)

        return img


    @staticmethod
    def laplacian(image: Image) -> np.ndarray:
        return ImageFilters.convolve(image, ImageFilters().laplacian)

    first_order_gradient = edge_detection
    second_order_gradient = laplacian
    