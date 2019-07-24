import numpy as np
import cv2
from pdb import set_trace

class SVM:
    svm = cv2.ml.SVM_create()
    TYPE = cv2.ml.SVM_C_SVC
    KERNEL = cv2.ml.SVM_LINEAR
    TERM_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    LAYOUT = cv2.ml.ROW_SAMPLE

    def train(self, data, labels):
        self.svm.setType(self.TYPE)
        self.svm.setKernel(self.KERNEL)
        self.svm.setTermCriteria(self.TERM_CRITERIA)
        print("Treinando a SVM ... isso pode levar alguns minutos/horas")
        self.svm.train(data.astype('float32'), self.LAYOUT, np.array(labels))
        print("Treinamento encerrado")
    
    def test(self, data):
        return self.svm.predict(data.reshape((1, -1)))[1]

    def save(self, filepath):
        self.svm.save(filepath)

    def load(self, filepath):
        self.svm = self.svm.load(filepath)
        if(cv2.isTrained(self.svm)):
            print("Modelo previamente treinado")