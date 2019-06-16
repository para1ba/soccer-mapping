import itertools as it
from descriptor import Descriptor
from svm import SVM

pos_desc = Descriptor('../dataset/pos/')
positive = pos_desc.describeAll()
pos_labels = it.repeat(1, len(positive))
neg_desc = Descriptor('../dataset/neg/')
negative = neg_desc.describeAll()
neg_labels = it.repeat(-1, len(negative))
svm = SVM()
svm.train(positive, pos_labels, negative, neg_labels)
#svm.test()