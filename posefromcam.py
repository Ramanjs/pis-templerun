# detect pose from webcam
import cv2
import mediapipe as mp
import numpy as np
import math
from pyautogui import hotkey
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

initial_height = [0, 0]
initialised = False
currentState = 'front'
width = 0
height = 0
cap = cv2.VideoCapture(0)

def getWeightedFaceMean(landmarks):
    total = 0.0
    for i in range(11):
        total += landmarks[i].x * landmarks[i].visibility

    total /= 10 
    return total

def getWeightedShoulderMean(landmarks):
    return (landmarks[11].x * landmarks[11].visibility + landmarks[12].x * landmarks[12].visibility) / 2

def isRight(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] 
    if (abs(left_shoulder.x - right_shoulder.x) <= 0.15 and
        (getWeightedFaceMean(landmarks) < getWeightedShoulderMean(landmarks))):
        return True
    return False

def isLeft(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] 
    if (abs(left_shoulder.x - right_shoulder.x) <= 0.15 and
        (getWeightedFaceMean(landmarks) > getWeightedShoulderMean(landmarks))):
        return True
    return False

def isFront(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] 
    face_mean = getWeightedFaceMean(landmarks)
    if (abs(left_shoulder.x - right_shoulder.x) > 0.18 and
        (face_mean < (left_shoulder.x * left_shoulder.visibility) and
         face_mean > (right_shoulder.x * right_shoulder.visibility))):
        return True
    return False

def isBack(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] 
    face_mean = getWeightedFaceMean(landmarks)
    if (abs(left_shoulder.x - right_shoulder.x) > 0.18 and
        (face_mean > (left_shoulder.x * left_shoulder.visibility) and
         face_mean < (right_shoulder.x * right_shoulder.visibility))):
        return True
    return False

def isTilting(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] 
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] 
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] 
    mean_shoulder = [(left_shoulder.x + right_shoulder.x) * width / 2 , (left_shoulder.y + right_shoulder.y) * height / 2]
    mean_hip = [(left_hip.x + right_hip.x) * width / 2 , (left_hip.y + right_hip.y) * height / 2]
    vertical = [mean_hip[0], mean_hip[1] - 5]

    a = np.array(mean_shoulder)
    b = np.array(mean_hip)
    c = np.array(vertical)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    print(angle)

    if (angle < 5):
        return False

    if (currentState == 'front'):
        if (mean_shoulder[0] < mean_hip[0]):
            return 'Tilt right'
        else:
            return 'Tilt left'
    elif (currentState == 'back'):
        if (mean_shoulder[0] < mean_hip[0]):
            return 'Tilt left'
        else:
            return 'Tilt right'
    elif (currentState == 'left'):
        if (mean_shoulder[2] < mean_hip[2]):
            return 'Tilt right'
        else:
            return 'Tilt left'
    elif (currentState == 'right'):
        if (mean_shoulder[2] < mean_hip[2]):
            return 'Tilt left'
        else:
            return 'Tilt right'

def isCrouching(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] 
    mean_height = [(left_shoulder.x + right_shoulder.x) / 2 , (left_shoulder.y + right_shoulder.y) / 2]
    distance = (((mean_height[0] - initial_height[0]) * width)**2 + ((mean_height[1] - initial_height[1]) * height)**2)**(0.5)
    if (mean_height[1] > initial_height[1] and distance > height / 8):
        return True
    return False

def getPrediction(landmarks):
    if (isRight(landmarks)):
        return 'right'
    if (isLeft(landmarks)):
        return 'left'
    if (isFront(landmarks)):
        return 'front'
    if (isBack(landmarks)):
        return 'back'

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        width, height = frame.shape[:2]
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        prediction = ''
        try:
            landmarks = results.pose_landmarks.landmark

            if (not initialised):
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] 
                initial_height = [(left_shoulder.x + right_shoulder.x) / 2 , (left_shoulder.y + right_shoulder.y) / 2]
                initialised = True

            prediction = getPrediction(landmarks)
            if prediction==None:
                prediction = currentState
            print(currentState,prediction)      
            if (currentState == 'front'):
                if (prediction == 'left'):
                    hotkey('left')
                    hotkey('win', 'shift', 'left')
                elif (prediction == 'right'):
                    hotkey('right')
                    hotkey('win', 'shift', 'right')
            elif (currentState == 'left'):
                if (prediction == 'back'):
                    hotkey('left')
                    hotkey('win', 'shift', 'left')
                elif (prediction == 'front'):
                    hotkey('right')
                    hotkey('win', 'shift', 'right')
            elif (currentState == 'back'):
                if (prediction == 'left'):
                    hotkey('right')
                    hotkey('win', 'shift', 'right')
                elif (prediction == 'right'):
                    hotkey('left')
                    hotkey('win', 'shift', 'left')
            elif (currentState == 'right'):
                if (prediction == 'front'):
                    hotkey('left')
                    hotkey('win', 'shift', 'left')
                elif (prediction == 'back'):
                    hotkey('right')
                    hotkey('win', 'shift', 'right')
            currentState = prediction

            if (isCrouching(landmarks)):
                print('crouch')
                hotkey('down')

            tilt = isTilting(landmarks)
            if (tilt):
                print(tilt)
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        )
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
