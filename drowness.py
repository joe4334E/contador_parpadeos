# Dr. Juan Zamora-Mora
# www.doczamora.com 

import cv2 as cv
import numpy as np
import mediapipe as mp

# vertical distance betwen min-max y-axis points. 
def open_len(arr):
    y_arr = []
    
    for _,y in arr:
        y_arr.append(y)
        
    min_y = min(y_arr)
    max_y = max(y_arr)
    
    return max_y - min_y

mp_face_mesh = mp.solutions.face_mesh

# A: location of the eye-landamarks in the facemesh collection
RIGHT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

# handle of the webcam
cap = cv.VideoCapture(0)

# Mediapipe parametes
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    # B: count how many frames the user seems to be going to nap (half closed eyes)
    drowsy_frames = 0
    
    # C: max height of each eye
    max_left = 0
    max_right = 0
    
    while True:

        # get every frame from the web-cam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the current frame and collect the image information
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        
        # D: collect the mediapipe results
        results = face_mesh.process(rgb_frame)
        
        # E: if mediapipe was able to find any landmanrks in the frame...
        if results.multi_face_landmarks:
            
            # F: collect all [x,y] pairs of all facial landamarks
            all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            # G: right and left eye landmarks
            right_eye = all_landmarks[RIGHT_EYE]
            left_eye = all_landmarks[LEFT_EYE]
            
            # H: draw only landmarks of the eyes over the image
            cv.polylines(frame, [left_eye], True, (0,255,0), 1, cv.LINE_AA)
            cv.polylines(frame, [right_eye], True, (0,255,0), 1, cv.LINE_AA) 
                        
            # I: estimate eye-height for each eye
            len_left = open_len(right_eye)
            len_right = open_len(left_eye)
            
            # J: keep highest distance of eye-height for each eye
            if len_left > max_left:
                max_left = len_left
                
            if len_right > max_right:
                max_right = len_right
            
            # print on screen the eye-height for each eye
            cv.putText(img=frame, text='Max: ' + str(max_left)  + ' Left Eye: ' + str(len_left), fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
            cv.putText(img=frame, text='Max: ' + str(max_right)  + ' Right Eye: ' + str(len_right), fontFace=0, org=(10, 50), fontScale=0.5, color=(0, 255, 0))
            
            # K: condition: if eyes are half-open the count.
            if (len_left <= int(max_left / 2) + 1 and len_right <= int(max_right / 2) + 1):
                drowsy_frames += 1
            else:
                drowsy_frames = 0
                
            # L: if count is above k, that means the person has drowsy eyes for more than k frames.
            if (drowsy_frames > 20):
                cv.putText(img=frame, text='ALERT', fontFace=0, org=(200, 300), fontScale=3, color=(0, 255, 0), thickness = 3)
                
            
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
            
cap.release()
cv.destroyAllWindows()
