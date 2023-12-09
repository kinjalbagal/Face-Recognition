from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import numpy as np

import imutils
from datetime import datetime
import mtcnn
import insightface

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(r'.\aug'))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
detector = mtcnn.MTCNN()

# loop over the image paths

for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    print(imagePath)
    name = imagePath.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    resized_image = cv2.resize(image, (600, 400))
    rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    boxe = detector.detect_faces(rgb)

    if boxe==[]:
        boxe = detector.detect_faces(rgb)
        print(boxe,"In")
        if boxe==[]:
            continue

    boxes=(boxe[0]["box"])
    boxes = [tuple(boxes)]

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
# print("[INFO] serializing encodings...")

data = {"encodings": knownEncodings, "names": knownNames}
print(data["names"])
f = open(r'.\encodings.pickle', "wb")
f.write(pickle.dumps(data))
f.close()

import pickle
inf = pickle.loads(open(r'.\encodings.pickle', "rb").read())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(inf["names"])

x=inf["encodings"]
y=labels
# import support vector classifier
# "Support Vector Classifier"
import sklearn
from sklearn.svm import SVC
clf = SVC(C=1.0,kernel='linear',probability=True)

clf.fit(x, y)
with open(r'C:\Users\jasmi\Desktop\PROJECTS\CO diploma\fin_file\model_new.pkl','wb') as f:
    pickle.dump(clf,f)
