import itertools as it
import cv2
from descriptor import Descriptor
from svm import SVM
import slidingwindow
import numpy as np
from os import path
import sys

print(sys.argv)
train = False
count = 1
name_video = '../sample-extractor/cutvideo.mp4'
if(len(sys.argv) >= 2):
    if(sys.argv[1] == "train"):
        train = True
    else:
        name_video = sys.argv[1]

if(not train):
    print("- CARREGANDO MODELO SVM EXISTENTE -")
    descriptor = Descriptor('')
    svm = SVM()
    svm.load("../models/trained_model.xml")
    vidcap = cv2.VideoCapture(name_video)
    success, image = vidcap.read()
    while success:
        for cut in slidingwindow.cut_frame(image):
            if cut[0].shape[0] == 128 and cut[0].shape[1] == 48:
                description = descriptor.describeImage(cut[0])
                result = int(list(svm.test(description))[0][0])
                print(result)
                if(result == 1):
                    cv2.imwrite("../figs/official/positive_" + str(count) + ".jpg", cut[0])
                    count += 1
            #cv2.imwrite("../figs/official/positive_" + str(count) + ".jpg", cut[0])
            #count += 1
        success, image = vidcap.read()

    '''
    positive_desc = Descriptor('../dataset/pos/for-test/')
    negative_desc = Descriptor('../dataset/neg/for-test/')
    test_positive = positive_desc.describeAll(0)
    test_negative = negative_desc.describeAll(0)
    print("ASSERÇÃO POSITIVA:")
    print(svm.countAssertion(svm.testAll(test_positive), 1), "%")
    print("ASSERÇÃO NEGATIVA:")
    print(svm.countAssertion(svm.testAll(test_negative), -1), "%")
    '''
else:
    '''
    print("- TREINANDO UM NOVO MODELO SVM -")
    pos_desc = Descriptor('../dataset/pos/for-train/')
    #positive = pos_desc.describeAll(60)
    positive = pos_desc.describeAll(0)
    pos_labels = list(it.repeat(1, len(positive)))
    neg_desc = Descriptor('../dataset/neg/for-train/')
    #negative = neg_desc.describeAll(240)
    negative = neg_desc.describeAll(0)
    neg_labels = list(it.repeat(-1, len(negative)))
    svm = SVM()
    svm.train(np.vstack((positive, negative)), list(pos_labels + neg_labels))
    svm.save("../models/trained_model.xml")
    '''
    pass
