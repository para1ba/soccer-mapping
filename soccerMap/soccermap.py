import cv2
import numpy as np
import random
from glob import glob
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from pdb import set_trace as pause
import os

## VIDEO INITIALIZING
name_video = '../sample-extractor/cutvideo.mp4'
images_path = '../dataset/images/'

## MODELS PATH_NAME
model_svm = '../models/svm_model.xml'
model_adaboost = '../models/adaboost_model.xml'

## HOG INITIALIZING
winSize = (32,96)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

## MODEL INITIALIZING
model = (None, 'None')

def breakLines(lines=20):
    for _ in range(lines):
        print('\n')

## RETURNS [(miniature, Y, X, Y + HEIGHT, X + WIDTH)]
def cut_frame(frame, window_height=128, window_width=48, increment=5):
    x_window = 0
    arr = []
    while (x_window + window_width) <= frame.shape[1]:
        y_window = 0
        while (y_window + window_height) <= frame.shape[0]:
            arr.append((frame[y_window : y_window + window_height, x_window : x_window + window_width], 
                       y_window, x_window, y_window + window_height, x_window + window_width))
            y_window += increment
        x_window += increment
    return arr

## DATASET INITIALIZING
neg_dirs = ["../dataset/neg/novo","../dataset/neg/negative"]
pos_dirs = ["../dataset/pos/cam_a","../dataset/pos/cam_b","../dataset/pos/positive"]

neg_files, pos_files, test_image_files = [], [], []
for path in neg_dirs: 
    neg_files += glob(path + "/*.jpg")
for path in pos_dirs: 
    pos_files += glob(path + "/*.jpg")
test_image_files += glob(images_path + "*.*")
all_files = pos_files + neg_files

while True:
    breakLines()
    print("=============== MENU ===============")
    print("   1. Treinar Modelo Usando AdaBoost")
    print("   2. Treinar Modelo Usando SVM")
    print("   3. Testar Usando Vídeo")
    print("   4. Testar Usando Imagens")
    print("   0. Sair")
    opt = input()
    breakLines()
    if opt == '1' or opt == '2':
        if (not os.path.exists(model_adaboost) and opt == '1') or (not os.path.exists(model_svm) and opt == '2'):
            print('Gerando descrições do conjunto de treinamento')
            data_train = []
            label_train = []
            data_test = []
            label_test = []
            test_size = 0.2
            for i, file in enumerate(all_files):
                #if i == 10:
                    #break
                I = cv2.imread(file)
                I = cv2.resize(I, winSize)
                H = hog.compute(I)
                cl = 1 if i < len(pos_files) else 0
                if random.random() < test_size:
                    data_test.append(H)
                    label_test.append(cl)
                else:
                    data_train.append(H)
                    label_train.append(cl) 
            data_train, data_test, label_train, label_test = [np.array(d).squeeze() for d in [data_train,data_test,label_train,label_test]]
            print("Dimensão final:",data_train.shape[1])
            print("Tamanho do conjunto de treinamento:", data_train.shape[0])
            print("Tamanho do conjunto de teste:",data_test.shape[0])
            print("Dados coletados!")
        if opt == '1':
            print("Treinando AdaBoost")
            model = (AdaBoostClassifier(n_estimators=200, random_state=0), 'AdaBoost')
            model[0].fit(data_train, label_train)
            pred = model[0].predict(data_test)
            #print("Erro teste:",np.abs(pred-label_test).mean())
            pred = model[0].predict(data_train)
            #print("Erro treinamento:",np.abs(pred-label_train).mean())
            #pause()
        elif opt == '2':
            if not os.path.exists(model_svm):
                model = (cv2.ml.SVM_create(), 'SVM')
                model[0].setKernel(cv2.ml.SVM_RBF)
                model[0].setType(cv2.ml.SVM_C_SVC)
                model[0].setC(2.5)
                model[0].setGamma(0.03375)
                model[0].train(data_train, cv2.ml.ROW_SAMPLE, label_train)
                #print("Numero de vetores suporte:",model[0].getSupportVectors().shape[0])
                pred = model[0].predict(data_test)[1]
                #print("Erro teste:",np.abs(pred-label_test).mean())
                pred = model[0].predict(data_train)[1]
                #print("Erro treinamento:",np.abs(pred-label_train).mean())
                model[0].save(model_svm)
            else:
                model = (cv2.ml.SVM_load(model_svm), 'SVM')
            
            #iris = datasets.load_iris()
            #X, y = iris.data, iris.target
            #model[0].fit(X, y)

        print('- TREINAMENTO CONCLUÍDO -')
    elif opt == '3':
        if model[0] == None:
            print('Treine o Modelo Antes de Testar')
        else:
            print('Analisando imagens do Vídeo')
            vidcap = cv2.VideoCapture(name_video)
            # success, image = vidcap.read()
            success, image = True, cv2.imread("../dataset/nHtLh.jpg")
            count = 1
            while success:
                #for scale in [2.0, 1.5, 1.0, 0.5]:
                for scale in [2.0, 1.75, 1.5, 1.25, 1.0]:
                    frame_scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                    miniatures = cut_frame(frame_scaled, window_height=128, window_width=48+16, increment=24)
                    for miniature in miniatures:
                        img = cv2.resize(miniature[0], hog.winSize)
                        description = hog.compute(img).T
                        if model[1] == 'SVM':
                            #result = model[0].predict(np.array(description))[1][0][0]
                            result = model[0].predict(np.array(description), flags=cv2.ml.StatModel_RAW_OUTPUT)[1][0][0]
                        elif model[1] == 'AdaBoost':
                            result = model[0].predict(description)
                        if(result < -0.1):
                            cv2.rectangle(frame_scaled,(miniature[2],miniature[1]),(miniature[4],miniature[3]),(0,0,255),3)
                    cv2.imwrite("../figs/official/frame_" + str(count) + ".jpg", frame_scaled)
                    print("ANALISADO! frame_" + str(count))
                    count += 1
                success, image = vidcap.read()
                break
    elif opt == '4':
        if model[0] == None:
            print('Treine o Modelo Antes de Testar')
        else:
            for file in test_image_files:
                image = cv2.imread(file)
                for scale in [2.0, 1.75, 1.5, 1.25, 1.0]:
                    frame_scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                    miniatures = cut_frame(frame_scaled, window_height=128, window_width=48+16, increment=24)
                    for miniature in miniatures:
                        img = cv2.resize(miniature[0], hog.winSize)
                        description = hog.compute(img).T
                        if model[1] == 'SVM':
                            result = model[0].predict(np.array(description), flags=cv2.ml.StatModel_RAW_OUTPUT)[1][0][0]
                        elif model[1] == 'AdaBoost':
                            result = model[0].predict(description)
                        if(result < -0.1):
                            cv2.rectangle(frame_scaled,(miniature[2],miniature[1]),(miniature[4],miniature[3]),(0,0,255),3)
                    cv2.imwrite("../figs/official/frame_" + str(count) + ".jpg", frame_scaled)
                    count += 1
    elif opt == '0':
        breakLines()
        break
    else:
        print('- OPÇÃO INVÁLIDA -')