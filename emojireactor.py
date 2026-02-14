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
    neutral_emoji = cv.imread("neutral.jpg")
    nerd_emoji = cv.imread("nerd.jpg")
    shush_emoji = cv.imread("shush.webp")
    freaky_emoji = cv.imread("tongue.jpg")
    mouth_open_emoji = cv.imread("mouthopen.jpg")

    #Check if emoji files are present
    if neutral_emoji is None:
        raise FileNotFoundError("neutral.jpg not found")
    if nerd_emoji is None:
        raise FileNotFoundError("nerd.jpg not found")
    if shush_emoji is None:
        raise FileNotFoundError("shush.webp not found")
    if freaky_emoji is None:
        raise FileNotFoundError("tongue.jpg not found")
    if mouth_open_emoji is None:
        raise FileNotFoundError("mouthopen.jpg not found")
    
    neutral_emoji = cv.resize(neutral_emoji, EMOJI_WINDOW_SIZE)
    nerd_emoji = cv.resize(nerd_emoji, EMOJI_WINDOW_SIZE)
    shush_emoji = cv.resize(shush_emoji, EMOJI_WINDOW_SIZE)
    freaky_emoji = cv.resize(freaky_emoji, EMOJI_WINDOW_SIZE)
    mouth_open_emoji = cv.resize(mouth_open_emoji, EMOJI_WINDOW_SIZE)
except Exception as e:
    print(f"error details: {e}")



#incase of error, shows a blank emoji
blank_emoji = np.zeros(EMOJI_WINDOW_SIZE, dtype=np.uint8)
emoji_to_display = neutral_emoji   


#MAIN LOGIC

cap = cv.VideoCapture(0)

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
        



        # init landmarks as None (prevents undefined variable errors)
        right_shoulder = right_wrist = None
        index_tip = wrist = None


        
        #mediapipe landmarks

        results_pose = pose.process(image_rgb)
        results_face = face_mesh.process(image_rgb)
        results_hands = hands.process(image_rgb)
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        else:
            right_shoulder = right_wrist = None
            
            
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0].landmark
            right_lip_corner = face_landmarks[61]
            left_lip_corner  = face_landmarks[291]
            eye_landmark = face_landmarks[33]

        if results_hands.multi_hand_landmarks:
            for hand_landmarks, hand_label in zip(results_hands.multi_hand_landmarks,
                                              results_hands.multi_handedness):
                label = hand_label.classification[0].label  
                if label == "Left":   
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    break   


        
        current_state = "relaxed"
        h, w , _ = frame.shape
        




        #Conditions
        if index_tip and results_face.multi_face_landmarks:
            lip_center = ((face_landmarks[13].x + face_landmarks[14].x) / 2,
            (face_landmarks[13].y + face_landmarks[14].y) / 2)
            dist = math.hypot(index_tip.x - lip_center[0], index_tip.y - lip_center[1])
            if dist < 0.1:
                emoji_to_display = shush_emoji
                current_state = "Shushed"

        if current_state != "Shushed":        
            if right_shoulder and right_wrist and index_tip:
                if (right_wrist.y < right_shoulder.y) and (index_tip.y < eye_landmark.y) and (eye_landmark.y < right_wrist.y):
                    #cv.putText(frame, "Raised up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    current_state = "Raised up"
        
        if current_state != "Raised up" and current_state != "Shushed":
            if results_face.multi_face_landmarks:
                dist = math.hypot(face_landmarks[13].x - face_landmarks[14].x, face_landmarks[13].y - face_landmarks[14].y)
                cx = int((face_landmarks[13].x + face_landmarks[14].x)/2 * w)
                cy = int(((face_landmarks[13].y + face_landmarks[14].y)/2 + 0.02) * h)
                if 0 <= cx < w and 0 <= cy < h:
                    pixel_color = frame[cy, cx]  # BGR
                    avg_red = int(pixel_color[2])  # R channel
                else:
                    avg_red = 0

                # Horizontal width (between lip corners)
                mouth_width = math.hypot(face_landmarks[61].x - face_landmarks[291].x,
                                        face_landmarks[61].y - face_landmarks[291].y)

                # Vertical opening (between upper and lower lips)
                mouth_height = math.hypot(face_landmarks[13].x - face_landmarks[14].x,
                                        face_landmarks[13].y - face_landmarks[14].y)

                aspect_ratio = mouth_height / mouth_width





                if dist > 0.05:
                    if avg_red > 120 and aspect_ratio > 0.25:
                        current_state = "freaky"
                    else:
                        current_state = "Mouth open"
                else:
                    current_state = "relaxed"
                                            
        
        #Condition to check for displaying the emoji
        if current_state == "Raised up":
            emoji_to_display = nerd_emoji
        elif current_state == "Shushed":
            emoji_to_display = shush_emoji
        elif current_state == "relaxed":
            emoji_to_display = neutral_emoji
        elif current_state == "Mouth open":
            emoji_to_display = mouth_open_emoji
        elif current_state == "freaky":
            emoji_to_display = freaky_emoji


            

        #mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        cv.imshow("frame",frame)
        cv.imshow('Emoji Output', emoji_to_display)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break


