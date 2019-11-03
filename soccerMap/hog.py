import cv2

class HOG():
    winSize = (32,96)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9

    def __init__(self):
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, 
                                     self.blockStride, self.cellSize,
                                     self.nbins)

    
