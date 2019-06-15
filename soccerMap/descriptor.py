from dataset import Dataset
import cv2

class Descriptor:
    def __init__(self, path):
        self.path = path
        self.dataset = Dataset(self.path)

    def describeAll(self):
        descriptions = []
        cell_size = (8, 8)
        block_size = (1, 1)
        nbins = 9

        for file in self.dataset.getPics():
            image = cv2.imread(file)
            for round in range(2):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // cell_size[1] * cell_size[1],
                                            gray.shape[0] // cell_size[0] * cell_size[0]),
                                            _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                            _blockStride=(cell_size[1], cell_size[0]),
                                            _cellSize=(cell_size[1], cell_size[0]),
                                            _nbins=nbins)
                n_cells = (gray.shape[0] // cell_size[0], gray.shape[1] // cell_size[1])
                hog_feats = hog.compute(gray).reshape(n_cells[1] - block_size[1] + 1,
                                                        n_cells[0] - block_size[0] + 1,
                                                        block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4))
                gradients = np.zeros((n_cells[0], n_cells[1], nbins))
                cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
                for off_y in range(block_size[0]):
                    for off_x in range(block_size[1]):
                        gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                        off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                        hog_feats[:, :, off_y, off_x, :]
                        cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                        off_x:n_cells[1] - block_size[1] + off_x + 1] += 1
                gradients /= cell_count
                descriptions.append(gradients)
                image = cv2.flip(image, 1)

        return descriptions
