from core.image import Image
import numpy as np

class Distance:
    # def __init__(self) -> None:
    #     pass

    @staticmethod
    def L_norm(img1: Image, img2: Image, n: int) -> float:
        return np.sum(np.abs(img1.data - img2.data)**n)**(1/n)
    
    @staticmethod
    def L_inf(img1: Image, img2: Image) -> float:
        return np.max(np.abs(img1.data - img2.data))
    
    @staticmethod
    def Manhattan(img1 :Image, img2: Image) -> float:
        return Distance.L_norm(img1, img2, 1)
    
    @staticmethod
    def Euclidean(img1: Image, img2: Image) -> float:
        return Distance.L_norm(img1, img2, 2)
    