import cv2 as cv
import os
import pickle
from sklearn.svm import SVC
from skimage.filters import gabor
import numpy as np

cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
haar_model = os.path.join(
    cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
svm = pickle.load(open('model9.sav', 'rb'))
haar_classifier = cv.CascadeClassifier(haar_model)
names = {0: 'Mask', 1: 'Without Mask'}

def kmeans(img):
    Z = img.reshape(-1)
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS +
                cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv.kmeans(
        Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def gabor_filter(img):
    return gabor(img, frequency=0.8)[0]

capture = cv.VideoCapture(0)

if not capture.isOpened():
    capture.open(0)


if capture.isOpened():
    while True:
        flag, img = capture.read()
        if flag:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.equalizeHist(img)
            faces = haar_classifier.detectMultiScale(img, minNeighbors = 5, minSize = (100,100))
            for x, y, w, h in faces:
                cv.rectangle(img, (x,y), (x+w, y+h), (122,50,200),2)
                face = img[y:y+h, x:x+w]
                face = cv.resize(face, (50, 50))
                face = gabor_filter(face)
                face = kmeans(face)
                face = face.reshape(1,-1)
                pred = svm.predict(face)[0]
                n = names[int(pred)]
                print(n)
            cv.imshow('test',img)
        if cv.waitKey(2) == 27:
            break

    capture.release()
    cv.destroyAllWindows()
