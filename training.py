import cv2
import numpy as np
import random

from glob import glob
from pdb import set_trace as pause

from sklearn.ensemble import AdaBoostClassifier

neg_dirs = ["dataset/neg/novo","dataset/neg/negative"]
pos_dirs = ["dataset/pos/cam_a","dataset/pos/cam_b","dataset/pos/positive"]
# pos_dirs = ["dataset/pos/positive"]

neg_files, pos_files = [], []
for path in neg_dirs: neg_files += glob(path + "/*.jpg")
for path in pos_dirs: pos_files += glob(path + "/*.jpg")


if __name__ == "__main__":

	test_size = .20 # 20% of the dataset

	winSize 			= (32,96)
	blockSize 			= (16,16)
	blockStride 		= (8,8)
	cellSize 			= (8,8)
	nbins 				= 9

	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

	all_files = pos_files + neg_files

	data_train,data_test   = [],[]
	label_train,label_test = [],[]

	for i,file in enumerate(all_files):

		I = cv2.imread(file)
		I = cv2.resize(I,winSize)
		H = hog.compute(I)
		cl = 1 if i < len(pos_files) else 0

		if random.random() < test_size:
			data_test.append(H)
			label_test.append(cl)
		else:
			data_train.append(H)
			label_train.append(cl) 

	data_train,data_test,label_train,label_test = [np.array(d).squeeze() for d in [data_train,data_test,label_train,label_test]]
	print("Dimensão final:",data_train.shape[1])
	print("Tamanho do conjunto de treinamento:",data_train.shape[0])
	print("Tamanho do conjunto de teste:",data_test.shape[0])
	print("Dados coletados")
	print("Treinando AdaBoost")
	model = AdaBoostClassifier(n_estimators=200, random_state=0)
	model.fit(data_train,label_train)

	pred = model.predict(data_test)
	print("Erro teste:",np.abs(pred-label_test).mean())

	pred = model.predict(data_train)
	print("Erro treinamento:",np.abs(pred-label_train).mean())

	print("Treinando SVM")
	model = cv2.ml.SVM_create()
	model.setKernel(cv2.ml.SVM_RBF)
	model.setType(cv2.ml.SVM_C_SVC)
	model.setC(2.5)
	model.setGamma(0.03375)

	# model.trainAuto(data_train, cv2.ml.ROW_SAMPLE, label_train)
	model.train(data_train, cv2.ml.ROW_SAMPLE, label_train)
	
	print("Numero de vetores suporte:",model.getSupportVectors().shape[0])

	pred = model.predict(data_test)[1]
	print("Erro teste:",np.abs(pred-label_test).mean())

	pred = model.predict(data_train)[1]
	print("Erro treinamento:",np.abs(pred-label_train).mean())

