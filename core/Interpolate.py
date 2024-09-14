from image import Image
import numpy as np


def handlergbMiddleware(func):
    def wrapper(image,factor,*args, **kwargs):
        if(len(image.data.shape) > 2):
            img = Image()
            img.data = np.zeros((image.data.shape[0]*factor, image.data.shape[1]*factor, image.data.shape[2]))
            for i in range(image.data.shape[2]):
                imgi = Image()
                imgi.data = image.data[:,:,i]
                img.data[:,:,i] = func(imgi, factor).data

            # normalize the image to be in the range [0, 255]
            img.data = (img.data - np.min(img.data)) / (np.max(img.data) - np.min(img.data))
            return img
        else:
            return func(image, factor)

    return wrapper


class Interpolation:
    def __init__(self):
        pass

    @staticmethod
    def decimation(image: Image, factor: int) -> Image:
        img = Image()
        img.data = image.data[::factor, ::factor]
        return img
    
    @handlergbMiddleware
    @staticmethod
    def nearest_neighbor(image: Image, factor: int) -> Image:
        img = Image()
        img.data = np.array([image.data[i//factor, j//factor] for i in range(image.data.shape[0]*factor) for j in range(image.data.shape[1]*factor)]).reshape(image.data.shape[0]*factor, image.data.shape[1]*factor)
        return img
    
    @handlergbMiddleware
    @staticmethod
    def bilinear(image: Image, factor: int) -> Image:
        img = Image()
        img.data = np.zeros((image.data.shape[0]*factor, image.data.shape[1]*factor))

        for i in range(img.data.shape[0]):
            for j in range(img.data.shape[1]):
                x = i/factor
                y = j/factor
                x1 = int(x)
                y1 = int(y)
                x2 = min(x1+1, image.data.shape[0]-1)
                y2 = min(y1+1, image.data.shape[1]-1)
                img.data[i, j] = (x2-x)*(y2-y)*image.data[x1, y1] + (x2-x)*(y-y1)*image.data[x1, y2] + (x-x1)*(y2-y)*image.data[x2, y1] + (x-x1)*(y-y1)*image.data[x2, y2]
                       
        return img

    @handlergbMiddleware
    @staticmethod
    def bicubic(image: Image, factor: int) -> Image:
        img = Image()
        # Initialize the output image data with zeros
        img.data = np.zeros((image.data.shape[0] * factor, image.data.shape[1] * factor))

        def cubic(x):
            abs_x = np.abs(x)
            abs_x2 = abs_x**2
            abs_x3 = abs_x**3
            return np.where(abs_x <= 1, (1.5 * abs_x3) - (2.5 * abs_x2) + 1,
                            np.where(abs_x <= 2, (-0.5 * abs_x3) + (2.5 * abs_x2) - (4 * abs_x) + 2, 0))

        for i in range(img.data.shape[0]):
            for j in range(img.data.shape[1]):
                x = i / factor
                y = j / factor
                x1 = int(x)
                y1 = int(y)
                x0 = max(x1 - 1, 0)
                x2 = min(x1 + 1, image.data.shape[0] - 1)
                x3 = min(x1 + 2, image.data.shape[0] - 1)
                y0 = max(y1 - 1, 0)
                y2 = min(y1 + 1, image.data.shape[1] - 1)
                y3 = min(y1 + 2, image.data.shape[1] - 1)
                x = x - x1
                y = y - y1
                A = np.array([[cubic(x + 1), cubic(x), cubic(x - 1), cubic(x - 2)]])
                B = np.array([[image.data[x0, y0], image.data[x0, y1], image.data[x0, y2], image.data[x0, y3]],
                              [image.data[x1, y0], image.data[x1, y1], image.data[x1, y2], image.data[x1, y3]],
                              [image.data[x2, y0], image.data[x2, y1], image.data[x2, y2], image.data[x2, y3]],
                              [image.data[x3, y0], image.data[x3, y1], image.data[x3, y2], image.data[x3, y3]]])
                img.data[i, j] = np.dot(np.dot(A, B), np.array([[cubic(y + 1)], [cubic(y)], [cubic(y - 1)], [cubic(y - 2)]]))

        return img
