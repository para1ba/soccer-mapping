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
        data = data.astype('float32')
        return self.svm.predict(data.reshape((1, -1)))[1]

    def testAll(self, allData):
        resp = []
        for data in allData:
            resp.append(int(self.test(data)[0][0]))
        return resp

    def countAssertion(self, tests, target=1):
        count = 0
        for test in tests:
            if(test == target):
                count+=1
        return (count*100)/len(tests)

    def save(self, filepath):
        self.svm.save(filepath)

    def load(self, filepath):
        self.svm = self.svm.load(filepath)