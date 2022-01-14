
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

message = ""

# Function to Extract useful Features from ROI by applying Different Filters and Techniques
def convert(frame):
    minValue = 70
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res


# Loading the Trained CNN Model
model = tf.keras.models.load_model('saving_model/myModel.h5')

#Labels
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Function to produce Label using CNN Model and feeding ROI
def prediction(image):
    img_array = keras.preprocessing.image.img_to_array(image)
    
    img_array = tf.expand_dims(img_array, 0) 
    
    # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)]

capture = cv2.VideoCapture(0)

while True:
    ret,frame=capture.read()
    frame=cv2.flip(frame,1)
    if ret==False:
        continue

    cv2.putText(frame, f'--- PREDICTION ---', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])

    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0) ,1)

    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (128, 128))

    #Extracing and Converting
    roi=convert(roi)
    #cv2.imshow("ROI", roi)

    #Display Prediction
    letter = prediction(roi)
    cv2.putText(frame, letter, (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 2)
    cv2.putText(frame, message, (10, 400), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 2)
    cv2.imshow("Frame", frame)

    #Detecting Key Interrupts
    interrupt = cv2.waitKey(10)
    
    if interrupt & 0xFF == 32: # space bar
        message += letter

    #if interrupt & 0xFF == 127: #backspace for mac
     #   message = message[0:-1]
    
    if interrupt & 0xFF == 47:
        message += ' ' 

    if interrupt & 0xFF == 8: #backspace for windows
        message = message[0:-1]

    if interrupt & 0xFF == 27: # esc key
        break

print("\n"  + message)

capture.release()
cv2.destroyAllWindows()