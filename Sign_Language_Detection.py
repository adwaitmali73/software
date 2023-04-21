#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# Keypoints Using MP Holistics

# In[4]:


mp_holistic = mp.solutions.holistic  # HOLISTICS MODEL
mp_drawing = mp.solutions.drawing_utils # DRAWING UTILITIES


# In[5]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make Prediction
    image.flags.writeable = True    # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB to BGR
    return image, results


# In[6]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Face Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Pose Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Left Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Right Hand Connections
    


# In[7]:


def draw_styled_landmarks(image, results):
    # Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color =(80,110,10), thickness = 1, circle_radius = 1), # Dot Color
                             mp_drawing.DrawingSpec(color =(80,256,121), thickness = 1, circle_radius = 1) # Line Color
                             )
                              
   # Pose Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color =(80,22,10), thickness = 2, circle_radius = 4), # Dot Color
                             mp_drawing.DrawingSpec(color =(80,44,121), thickness =2, circle_radius = 2) # Line Color
                             )  
    # Left Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color =(121,22,76), thickness = 2, circle_radius = 4), # Dot Color
                             mp_drawing.DrawingSpec(color =(121,44,250), thickness = 2, circle_radius =2) # Line Color
                             ) 
     # Right Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color =(245,117,66), thickness = 2, circle_radius = 4), # Dot Color
                             mp_drawing.DrawingSpec(color =(245,66,230), thickness = 2, circle_radius = 2) # Line Color
                             ) 
   
    


# In[9]:


cap = cv2.VideoCapture(0)
# Set mediapipe model

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
    
        # Read feed
        ret, frame = cap.read()
    
        # Make Detection
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        #Draw landmarks
        draw_styled_landmarks(image, results)
        
    
        #Show to Screen
        cv2.imshow("feed", image)
    
        # Break the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[11]:


results.pose_landmarks


# 3> Extract Keypoints Values

# In[17]:


pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)


# In[33]:


pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
face  = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)


# In[36]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face  = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
    


# In[ ]:




