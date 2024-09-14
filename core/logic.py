from core.image import Image
import numpy as np
import enum

class Binary_Operation(enum.Enum):
    ADD = lambda x, y: x + y
    SUB = lambda x, y: x - y
    MUL = lambda x, y: x * y
    DIV = lambda x, y: x / y
    AND = lambda x, y: x and y
    OR = lambda x, y: x or y
    XOR = lambda x, y: x ^ y

class Unary_Operation(enum.Enum):
    NOT = lambda x: not x


class BinaryImage(Image):

    def __init__(self, img: Image , threshold: float) -> None:
        super().__init__()
        self.data = img.data
        self.threshold = threshold
        self.binarize()
    
    def binarize(self) -> None:
        self.data = (self.data > self.threshold)

    
    def apply(self, image2: Image, operation: Binary_Operation) -> Image:
        img = Image()

        img.data = np.vectorize(operation)(self.data, image2.data)

        return img
    
    def apply_unary(self, operation: Unary_Operation) -> Image:
        img = Image()

        img.data = np.vectorize(operation)(self.data)

        return img