#===================================================================================================#
#                                      Lost Child Simulation                                        #
#    Last Modification: 30.07.2018                                         Mauricio Fadel Argerich  #
#===================================================================================================#

import glob
import os, sys
import cv2
import types
import time

sys.path.append('../utils/')
from util import detect_faces, TRAINING_FOLDER, RECOGNIZER_MODEL_CHEAP, RECOGNIZER_MODEL_EXPENSIVE, RECOGNIZER_MODEL_MIDDLE, CLASSIFIER_FOLDER, TESTING_SIMULATOR_FOLDER

from profiler import profile
from entities import MFAIO, MFAFunction, MFAProfileData

# This is needed because Profiler prints in results the str(param_value) for each
# parameter. Since CascadeClassifier and FaceRecognizer objects from OpenCV2 don't
# have this, we implement it in order to print something meaningful.
class Wrapper(object):
    str_id = ''

    def __init__(self, str_id, obj):
        self._wrapped_obj = obj
        self.str_id = str_id

    def __str__(self):
        return self.str_id

    def __getattr__(self, attr):
        if hasattr(self._wrapped_obj, attr):
            attr_value = getattr(self._wrapped_obj, attr)

            if isinstance(attr_value, types.MethodType):
                def callable(*args, **kwargs):
                    return attr_value(*args, **kwargs)
                return callable
            else:
                return attr_value
        else:
            raise AttributeError

#================================== FUNCTIONS ==================================#
# wrapper for detect_faces in utils
def detect_faces_sim(img, face_cascade, scale_factor, resize):
    faces_to_recognize, pos = detect_faces(img,
                                           face_cascade = face_cascade,
                                           scale_factor = scale_factor,
                                           resize = resize)
    return faces_to_recognize

def recognize_faces(faces_to_recognize, face_recognizer):
    res = False
    for i in range(len(faces_to_recognize)):
        p, c = face_recognizer.predict(faces_to_recognize[i])
        if p != -1:
            res = True
            break

    return res

lbp_cascade = cv2.CascadeClassifier(CLASSIFIER_FOLDER + 'lbpcascade_frontalface.xml')
lbp_wrapper = Wrapper('lbp', lbp_cascade)
haar_cascade = cv2.CascadeClassifier(CLASSIFIER_FOLDER + 'haarcascade_frontalface_default.xml')
haar_wrapper = Wrapper('haar', haar_cascade)

f_detect_faces = MFAFunction(function = detect_faces_sim, params = {'face_cascade': {lbp_wrapper:3, haar_wrapper:10},
                                                                'scale_factor': {1.2:10, 1.33:4},
                                                                'resize': {(120,120):10, (100,100):7}})

face_recognizer_expensive = cv2.face.LBPHFaceRecognizer_create()
face_recognizer_expensive.read(RECOGNIZER_MODEL_EXPENSIVE)
fre_wrapper = Wrapper('expensive', face_recognizer_expensive)
face_recognizer_middle = cv2.face.FisherFaceRecognizer_create()
face_recognizer_middle.read(RECOGNIZER_MODEL_MIDDLE)
frm_wrapper = Wrapper('middle', face_recognizer_middle)
face_recognizer_cheap = cv2.face.FisherFaceRecognizer_create()
face_recognizer_cheap.read(RECOGNIZER_MODEL_CHEAP)
frc_wrapper = Wrapper('cheap', face_recognizer_cheap)

f_recognize_faces = MFAFunction(function = recognize_faces, params = {'face_recognizer': {fre_wrapper:10,
                                                                                          frm_wrapper:5,
                                                                                          frc_wrapper:3}})

#====================================== IO =====================================#
testset = sorted(glob.glob(TESTING_SIMULATOR_FOLDER + '*.png'))
pipeline_inputs = []
for img_path in testset:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    pipeline_inputs.append(MFAIO(io_id=img_path[img_path.rfind(os.sep) + 1:],
                                   io_format='png',
                                   io_size=os.path.getsize(img_path),
                                   io_value=img))

#=================================== SIMULATE ==================================#
# The first function takes the input from pipeline_inputs, the next ones, take
# as input the output of the previous function, i.e.: f_recognize_faces(f_detect_faces(p_input)).
# The output of the last function will be save to results file.
pipeline = [f_detect_faces, f_recognize_faces]
profile('pc_nec', pipeline, pipeline_inputs, 5)