# importing necessary libraries
import numpy as np
import cv2  
from keras.models import load_model


# loading your gender classification model
my_model = load_model('gender_classification.h5')

# to view your keras model architecture
#print(my_model.summary())

# to know the format of input to your CNN model
#print("input: ", my_model.inputs)
# ---> input:  [<tf.Tensor 'conv2d_1_input:0' shape=(?, 64, 64, 3) dtype=float32>]

# to view the format of output by your CNN model
#print("output: ", my_model.outputs)
# ---> output:  [<tf.Tensor 'activation_1/Sigmoid:0' shape=(?, 1) dtype=float32>]

# you can download cascade features from https://github.com/opencv/opencv/ under data folder
# for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# capture frames from a camera 
cap = cv2.VideoCapture(0)


# # loop runs if capturing has been initialized. 
while True:  
  
    # reads frames from a camera 
    ret, image = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    # two parameter ScaleFactor and minNeighbors
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)  

        # cropping only the face part
        crop_face = image[y:y+h, x:x+w]

        # resizing to size that fits into your model architecture
        crop_face = cv2.resize(crop_face, (64,64))

        # expanding dimension according to your model input dimension requirement
        crop_face = np.expand_dims(crop_face, axis = 0)

        # prediction each face using your deep learning model
        prediction = my_model.predict(crop_face)
        print(prediction)
        if (prediction[0][0] >= 0.5):
            gender = 'female'
        else:
            gender = 'male'

        # putting text on video frame    
        cv2.putText(image, gender,(x, y-10),0, 5e-3 * 200, (0,255,0),2)        
    
    # Display an image in a window 
    cv2.namedWindow('classification', cv2.WINDOW_NORMAL)
    cv2.imshow('classification', image)
    
    # Wait for Esc key to stop 
    k = cv2.waitKey(2) & 0xff
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()