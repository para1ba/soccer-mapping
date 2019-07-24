import itertools as it
import cv2
from descriptor import Descriptor
from svm import SVM
import numpy as np
from pdb import set_trace

pos_desc = Descriptor('../dataset/pos/')
positive = pos_desc.describeAll(60)
pos_labels = list(it.repeat(1, len(positive)))
neg_desc = Descriptor('../dataset/neg/negative/')
negative = neg_desc.describeAll(240)
neg_labels = list(it.repeat(-1, len(negative)))
svm = SVM()
svm.train(np.vstack((positive, negative)), list(pos_labels + neg_labels))
svm.save('/home/para1ba/development/soccer-mapping/models/trained_model.xml')
image = cv2.imread("/home/para1ba/development/soccer-mapping/dataset/pos/positive/player1527.jpg")
desc = pos_desc.describeImage(image)
#set_trace()
print(svm.test(desc.astype('float32')))

'''
svm = SVM()
svm.load('/home/para1ba/development/soccer-mapping/models/trained_model.xml')
image = cv2.imread("/home/para1ba/development/soccer-mapping/dataset/pos/positive/player1527.jpg")
print(svm.test(image))
'''