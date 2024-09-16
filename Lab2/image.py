import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Image:
    def __init__(self, filename=None) -> None:

        if filename is not None:
            self.load(filename)
            self.type = filename.split('.')[-1]
            self.filename = filename

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        self.data = mpimg.imread(filename)

    def show(self) -> None:
        print(self.data)

    def _repr_png_(self) -> None:
        return self.data
    
    def display(self) -> None:
        if len(self.data.shape) == 2:  # Grayscale image
            plt.imshow(self.data, cmap='gray')
        elif len(self.data.shape) == 3 and self.data.shape[2] == 3:  # RGB image
            plt.imshow(self.data)
        elif len(self.data.shape) == 3 and self.data.shape[2] == 4:  # RGBA image
            plt.imshow(self.data)
        else:
            raise ValueError("Unsupported image format")
        plt.axis('off')  # Hide the axis
        plt.show()


class ImageProperties:
    @staticmethod
    def brightness(image: Image) -> float:
        return np.mean(image.data)
    
    @staticmethod
    def contrast(image: Image) -> float:
        return np.std(image.data)
    
    @staticmethod
    def histogram(image: Image) -> None:
        plt.hist(image.data.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        plt.show()

    @staticmethod
    def range(image: Image) -> tuple:
        return (np.min(image.data), np.max(image.data))
    
    @staticmethod
    def aspect_ratio(image: Image) -> tuple:
        return (image.data.shape[0], image.data.shape[1])
    
    @staticmethod
    def hue(image: Image) -> float:
        pass

    def saturation(image: Image) -> float:
        pass

    def value(image: Image) -> float:
        pass

    @staticmethod
    def std_deviation(image: Image) -> float:
        return np.std(image.data)
    
    @staticmethod
    def skewness(image: Image) -> float:
        return np.mean((image.data - np.mean(image.data))**3) / (np.std(image.data)**3)
    

    @staticmethod
    def summary(image: Image) -> dict:
        return {
            'filename': image.filename if hasattr(image, 'filename') else None,
            'image_tensor shape': image.data.shape,
            'brightness': ImageProperties.brightness(image),
            'contrast': ImageProperties.contrast(image),
            'range': ImageProperties.range(image),
            'aspect_ratio': ImageProperties.aspect_ratio(image),
            'std_deviation': ImageProperties.std_deviation(image),
            'skewness': ImageProperties.skewness(image)
        }
    
    @staticmethod
    def histogram(image: Image) -> None:
        range_ = ImageProperties.range(image)


        if(len(image.data.shape) == 3):
            hist, bins = np.histogram(image.data[:,:,0].ravel(), bins=256, range=range_)
            plt.plot(hist, color='red')
            hist, bins = np.histogram(image.data[:,:,1].ravel(), bins=256, range=range_)
            plt.plot(hist, color='green')
            hist, bins = np.histogram(image.data[:,:,2].ravel(), bins=256, range=range_)
            plt.plot(hist, color='blue')
        else:
            hist, bins = np.histogram(image.data.ravel(), bins=256, range=range_)
            plt.plot(hist, color='white')

    

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
    def average(image: Image) -> np.ndarray:
        return ImageFilters.convolve(image, np.ones((3, 3)) / 9)

    @staticmethod
    def laplacian(image: Image) -> np.ndarray:
        return ImageFilters.convolve(image, ImageFilters().laplacian)
    

    
    

    
