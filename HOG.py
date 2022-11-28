import cv2 as cv
import os
import pickle
from sklearn.svm import SVC
from skimage.filters import rank, sobel
import numpy as np
from skimage.morphology import disk
from skimage.feature import hog, local_binary_pattern, ORB


cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
haar_model = os.path.join(
    cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
svm = pickle.load(open('model16.sav', 'rb'))
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

def describe(image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = local_binary_pattern(image, 24,
			8, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, 24 + 3),
			range=(0, 24 + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist

def gabor_filter(img):
    return gabor(img, frequency=0.8)[0]

capture = cv.VideoCapture(1)

if not capture.isOpened():
    capture.open(1)


if capture.isOpened():
    while True:
        flag, img = capture.read()
        if flag:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = haar_classifier.detectMultiScale(img, minNeighbors = 5, minSize = (200,200))
            for x, y, w, h in faces:
                cv.rectangle(img, (x,y), (x+w, y+h), (122,50,200),2)
                face = img[y:y+h, x:x+w]
                face = cv.resize(face, (50, 50)) 
                face = rank.mean_bilateral(
                    face, footprint=disk(2), s0=50, s1=50
                )
                face = kmeans(face)
                fd = hog(face, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2))
                features = fd
                features = features.reshape(1,-1)
                pred = svm.predict(features)[0]
                n = names[int(pred)]
                cv.putText(img, n, (x+50,y), cv.FONT_HERSHEY_SIMPLEX ,1, (255,255,0) ,2, cv.LINE_AA)

                print(n)
            cv.imshow('test',img)


        if cv.waitKey(2) == 27:
            break

    capture.release()
    cv.destroyAllWindows()
