from core.image import Image
import matplotlib.pyplot as plt
import numpy as np

class Histogram:
    def __init__(self,img : Image, bins: int = 256) -> None:
        self.image = img
        self.bins = bins
        self.NM = self.image.data.shape[0]*self.image.data.shape[1]
        self.histogram = self.calculate_histogram()

    def calculate_histogram(self) -> np.ndarray:
        hist = np.zeros(self.bins)
        for i in range(self.image.data.shape[0]):
            for j in range(self.image.data.shape[1]):
                hist[self.image.data[i,j] ] += 1
        # return hist/self.NM
        return hist
    
    def cdf(self) -> np.ndarray:
        cdf = np.zeros(self.bins)
        cdf[0] = self.histogram[0]
        for i in range(1,self.bins):
            cdf[i] = cdf[i-1] + self.histogram[i]
        return cdf
    
    def inv_cdf(self) -> np.ndarray:
        cdf = self.cdf()
        cdf_min = cdf[0]
        inv_cdf = np.zeros(self.bins)
        for i in range(self.bins):
            inv_cdf[i] = round(((cdf[i] - cdf_min)/(self.image.data.shape[0]*self.image.data.shape[1] - cdf_min))*(self.bins-1))
        return inv_cdf

    def equalize(self) -> Image:
        inv_cdf = self.inv_cdf()

        img = Image()
        img.data = self.image.data.copy()
        for i in range(img.data.shape[0]):
            for j in range(img.data.shape[1]):
                img.data[i,j] = inv_cdf[img.data[i,j]]
        return img
    
    def plot(self):
        #plot histogram as bar chart
        plt.bar(range(self.bins), self.histogram)
        plt.show()
        return None
    
    def plot_cdf(self):
        #plot histogram as bar chart
        plt.bar(range(self.bins), self.cdf())
        plt.show()
        return None
    
    def match(self, target: Image) -> Image:
        target_hist = Histogram(target)

        target_inv_cdf = target_hist.inv_cdf()

        target_inv_cdf_inv = np.zeros(target_inv_cdf.shape)


        for i in range(target_inv_cdf.shape[0]):
            target_inv_cdf_inv[int(target_inv_cdf[i])] = i

        for i in range(1,target_inv_cdf_inv.shape[0]):
            if target_inv_cdf_inv[i] == 0:
                target_inv_cdf_inv[i] = target_inv_cdf_inv[i-1]

        source_inv_cdf = self.inv_cdf()
        source_inv_cdf = np.int32(source_inv_cdf)

        img = Image()
        img.data = self.image.data.copy()
        for i in range(img.data.shape[0]):
            for j in range(img.data.shape[1]):
                img.data[i,j] = target_inv_cdf_inv[source_inv_cdf[img.data[i,j]]]
        return img
    
