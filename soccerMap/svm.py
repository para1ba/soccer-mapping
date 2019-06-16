import numpy as np
import cv2

class SVM:
    svm = cv2.ml.SVM_create()
    TYPE = cv2.ml.SVM_C_SVC
    KERNEL = cv2.ml.SVM_LINEAR
    TERM_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    LAYOUT = cv2.ml.ROW_SAMPLE

    def train(self, data, labels):
        self.svm.setType(TYPE)
        self.svm.setKernel(KERNEL)
        self.svm.setTermCriteria(TERM_CRITERIA)
        self.svm.train(np.matrix(data), LAYOUT, np.array(labels))

    def train(self, pos_data, pos_labels, neg_data, neg_labels):
        self.train(np.vstack(np.matrix(pos_data), np.matrix(neg_data)), pos_labels + neg_labels)
    
    def test(self, data):
        result = self.svm(data)[1]

        return result