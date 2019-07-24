from dataset import Dataset
import numpy as np
import cv2

class Descriptor:
    cell_size = (8, 8)
    block_size = (1, 1)
    nbins = 9

    def __init__(self, path):
        self.path = path
        self.dataset = Dataset(self.path)

    def describeAll(self):
        return self.describeAll(self, 0)

    def describeAll(self, sample_limit):
        descriptions = []
        samples = 0

        for file in self.dataset.getPics():
            if samples == sample_limit and sample_limit != 0:
                break
            print("Descrevendo imagem: ", file)
            image = cv2.imread(file, 0)
            samples = samples + 1
            for round in range(2):
                gradients = self.describeImage(image)
                descriptions.append(gradients)
                image = cv2.flip(image, 1)

        return descriptions

    def describeImage(self, image):
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image
        hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // self.cell_size[1] * self.cell_size[1],
                                    gray.shape[0] // self.cell_size[0] * self.cell_size[0]),
                                    _blockSize=(self.block_size[1] * self.cell_size[1],
                                    self.block_size[0] * self.cell_size[0]),
                                    _blockStride=(self.cell_size[1], self.cell_size[0]),
                                    _cellSize=(self.cell_size[1], self.cell_size[0]),
                                    _nbins=self.nbins)
        n_cells = (gray.shape[0] // self.cell_size[0], gray.shape[1] // self.cell_size[1])
        hog_feats = hog.compute(gray).reshape(n_cells[1] - self.block_size[1] + 1,
                                                n_cells[0] - self.block_size[0] + 1,
                                                self.block_size[0], self.block_size[1], self.nbins).transpose((1, 0, 2, 3, 4))
        gradients = np.zeros((n_cells[0], n_cells[1], self.nbins))
        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
        for off_y in range(self.block_size[0]):
            for off_x in range(self.block_size[1]):
                gradients[off_y:n_cells[0] - self.block_size[0] + off_y + 1,
                off_x:n_cells[1] - self.block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - self.block_size[0] + off_y + 1,
                off_x:n_cells[1] - self.block_size[1] + off_x + 1] += 1
        gradients /= cell_count

        return np.array(gradients).flatten()
