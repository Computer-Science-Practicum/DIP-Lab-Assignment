from core.image import Image
import numpy as np

def matrix_inverse_2d(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a 2x2 matrix.
    """
    a, b, c, d = matrix.flatten()
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]]) / det


class LinearTransformation:
    """
    Linear transformation of image data.
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def transform(image: Image, matrix: np.ndarray) -> Image:
        if(type(matrix)==list):
            matrix = np.array(matrix)

        img = Image()
        img.data = np.zeros(image.data.shape)
        matrix_ = matrix_inverse_2d(matrix)
        for i in range(image.data.shape[0]):
            for j in range(image.data.shape[1]):
                x, y = np.dot(matrix_, np.array([i, j]))
                if 0 <= x < image.data.shape[0] and 0 <= y < image.data.shape[1]:
                    img.data[i, j] = image.data[int(x), int(y)]

        return img

    @staticmethod
    def scale(image: Image, factor: float) -> Image:
        return LinearTransformation.transform(image, np.array([[factor, 0], [0, factor]]))
    
    @staticmethod
    def rotate(image: Image, angle: float) -> Image:
        angle = np.radians(angle)
        return LinearTransformation.transform(image, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
    
    @staticmethod
    def shear(image: Image, factorx: float, factory: float) -> Image:
        return LinearTransformation.transform(image, np.array([[1, factorx], [factory, 1]]))
    
    @staticmethod
    def translate(image: Image, x: int, y: int) -> Image:
        img = Image()
        img.data = np.zeros(image.data.shape)
        for i in range(image.data.shape[0]):
            for j in range(image.data.shape[1]):
                if 0 <= i + x < image.data.shape[0] and 0 <= j + y < image.data.shape[1]:
                    img.data[i + x, j + y] = image.data[i, j]

        return img