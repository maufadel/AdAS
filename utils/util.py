import cv2
import glob
import numpy as np

# Paths -----------------------------------------------------------------------------------------------------
TRAINING_FOLDER = '../data/train/'
TESTING_FOLDER = '../data/test/'
TESTING_SIMULATOR_FOLDER = '../data/simulator/'
CLASSIFIER_FOLDER = '../utils/'
RECOGNIZER_MODEL_EXPENSIVE = '../utils/model_expensive.xml'
RECOGNIZER_MODEL_MIDDLE = '../utils/model_middle.xml'
RECOGNIZER_MODEL_CHEAP = '../utils/model_cheap.xml'
#------------------------------------------------------------------------------------------------------------

def detect_faces(img, face_cascade, scale_factor = 1.2, resize = None):
    # Convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier
    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=2)

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # images will contain only the faces images.
    images = []
    for f in faces:
        (x, y, w, h) = f
        face = gray[y:y + w, x:x + h]
        if resize:
           face = cv2.resize(face, resize)
        images.append(face)

    return images, faces

def detect_faces_gpu(img, face_cascade, scale_factor = 1.2, resize = None):
    # Convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier
    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=5)

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # images will contain only the faces images.
    images = []
    for f in faces:
        (x, y, w, h) = f
        crop = cv2.UMat(gray, [y, y+w], [x, x+h])
        images.append(crop)

    return images, faces
