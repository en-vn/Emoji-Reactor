import cv2 as cv
import mediapipe as mp
import numpy as np
import math

#Initialise mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

WINDOW_WIDTH = 480
WINDOW_HEIGHT = 480
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)


#reading emoji
try:
    neutral_emoji = cv.imread("neutralV2.png")
    raised_eyebrow_emoji = cv.imread("raised_eyebrow.png")
    thumbs_up_emoji = cv.imread("thumbsup.png")
    thumbs_down_emoji = cv.imread("thumbsdown.png")


    if neutral_emoji is None or raised_eyebrow_emoji is None or thumbs_up_emoji is None or thumbs_down_emoji is None:
        raise FileNotFoundError("Emoji files not found")
    


    neutral_emoji = cv.resize(neutral_emoji, EMOJI_WINDOW_SIZE)
    raised_eyebrow_emoji = cv.resize(raised_eyebrow_emoji, EMOJI_WINDOW_SIZE)
    thumbs_up_emoji = cv.resize(thumbs_up_emoji, EMOJI_WINDOW_SIZE)
    thumbs_down_emoji = cv.resize(thumbs_down_emoji, EMOJI_WINDOW_SIZE)
except Exception as e:
    print(e)

cap = cv.VideoCapture(0)


emoji_to_display = neutral_emoji
    
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
            mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv.flip(frame, 1)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False


        results_pose = pose.process(image_rgb)
        results_face = face_mesh.process(image_rgb)
        results_hands = hands.process(image_rgb)


        #mediapipe landmarks
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0].landmark
            brow = face_landmarks[296]
            eye = face_landmarks[386]

            

            
        #conditions
        dist = eye.y - brow.y
        if dist > 0.045:
            emoji_to_display = raised_eyebrow_emoji
        else:
            emoji_to_display = neutral_emoji





        cv.imshow("frame",frame)
        cv.imshow('Emoji Output', emoji_to_display)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break