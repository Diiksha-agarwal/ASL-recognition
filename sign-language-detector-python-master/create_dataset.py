import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'asl detector/sign-language-detector-python-master/data1'
data = pd.DataFrame(columns=['x0','x1','x2','x3', 'x4','x5','x6', 'x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20', 'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12','y13','y14','y15','y16','y17','y18','y19', 'y20',"class"])
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x_ = []
        y_ = []
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                new_row = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    new_row.extend([x- min(x_), y - min(y_)])
                new_row.append(dir_)
                data = data.append(pd.Series(new_row, index=data.columns), ignore_index=True)
                

# print(data.head())
data.to_csv('keypoints.csv',index=False)
