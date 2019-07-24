#fonte: https://www.youtube.com/watch?v=ou7SOV2xJ6k

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

#from pdb import set_trace as pause

cell_size = (8, 8)
block_size = (1, 1)
nbins = 9
first = True
positives = []
#for file in glob.glob("/home/para1ba/development/TCC/dataset/**/*.bmp", recursive=True):
# TO 
for file in glob.glob("./dataset/pos/cam_a/*.*", recursive=True):
    print("Analisando Imagem : " + file)
    image = cv2.imread(file)
    for i in range(2):
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
        positives.append(gradients)
        if(first):
            all_gradients = gradients
            first = False
        all_gradients += gradients
        image = cv2.flip(image, 1)
all_gradients /= len(list(glob.glob("./dataset/pos/cam_a/*.*", recursive=True)))
fig = plt.figure(num="Descrição HOG", figsize=(9, 13))
columns = 3
rows = 3
ax = []
for i in range(9):
    img = all_gradients[:, :, i]
    ax.append(fig.add_subplot(rows, columns, i+1))
    if(i==0):
        ax[-1].set_title("Ângulo:" + str(i * 20) + "/" + "360" + " e " + str(i * 20 + 180) + " graus")
    else:
        ax[-1].set_title("Ângulo:"+str(i*20) + " e "+ str(i*20+180) +" graus")
    plt.imshow(img, alpha=0.75)
plt.subplots_adjust(wspace = 0.2, hspace = 0.4)
plt.suptitle("Descrição HOG")
#cv2.imshow("antes", image)
plt.savefig('demo.png', bbox_inches='tight')
#plt.show()  # finally, render the plot
#cv2.waitKey(0)
#cv2.destroyAllWindows()
