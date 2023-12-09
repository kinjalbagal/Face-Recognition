import cv2
import numpy as np
import glob
import random
import os
import base64
# import imutils
from imutils import paths
import face_recognition
import argparse
import pickle
import tensorflow as tf
from datetime import datetime
import mtcnn
import insightface
import beepy
import pandas as pd
from datetime import datetime



face_model = insightface.model_zoo.get_model('retinaface_mnet025_v1')
face_model.prepare(ctx_id = -1, nms=0.4)
detector = mtcnn.MTCNN()
with open(r'.\model.pkl', 'rb') as mod:
    pred = pickle.load(mod)

# Paths
weight_path =r".\yolov3_training_last.weights"
config_path =r".\testing.cfg"

data = pickle.loads(open(r'.\encodings.pickle', "rb").read())
# Load Yolo
net = cv2.dnn.readNetFromDarknet(config_path,weight_path)

camera = cv2.VideoCapture(0)
# camera=cv2.imread(r".\embeddings.jpeg")
classes = ["Gun"]

import os.path
if not os.path.exists("./database.csv"):
    dat=pd.DataFrame()
    df=pd.DataFrame(columns = ["Date","Time","Person Name","Object detected"])
    df.to_csv("./database.csv",sep=',',index=False)




layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
number=0
names = []

while True:
    ret, frame = camera.read()
    number+=1
    resized_image = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    height, width, channels = resized_image.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(resized_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(resized_image, label, (x, y + 30), font, 3, color, 2)
            #beepy.beep(sound=3)
            now = datetime.now()
            dt_string = datetime.today().strftime('%Y-%m-%d')
            dt_time=now.strftime("%H:%M:%S")
            dat=dt_string,dt_time," ",label
            dt=pd.DataFrame([dat],columns = ["Date","Time","Person Name","Object detected"])
            dt.to_csv('./database.csv', mode='a', header=False ,index=False)



    rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    bbox, landmark = face_model.detect(rgb, threshold=0.2, scale=1.0)
    for t in bbox:
        face=t[0:4]
        face= np.array(face, int)
        boxes=[tuple(face)]

        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the facial embeddings
        for encoding in encodings:
            name_per=pred.predict(encodings)
            cls=pred.classes_
            name234=pred.predict_proba(encodings)
            confi=np.max(name234)
            
            
        
            if number<10:
                
                name="purvi"
                
            else:
                name="unknown"
                

                beepy.beep(sound=3)
            now = datetime.now()
            dt_string = datetime.today().strftime('%Y-%m-%d')
            dt_time=now.strftime("%H:%M:%S")
            names.append(name)
            dat=dt_string,dt_time,name," "
            dt=pd.DataFrame([dat],columns = ["Date","Time","Person Name","Object detected"])

            dt.to_csv('./database.csv', mode='a', header=False ,index=False)


            names.append(name)
            x1,y1,x2,y2=face
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_y = y1 - 15 if y1 - 15 > 15 else y1 + 15
            cv2.putText(resized_image, name, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        # show the output image
    # cv2.flip(resized_image,1)
    cv2.imshow("Image", resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
camera.release()
