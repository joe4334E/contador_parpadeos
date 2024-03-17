import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import time
import sqlite3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def open_len(arr):
    y_arr = []
    for _, y in arr:
        y_arr.append(y)
    min_y = min(y_arr)
    max_y = max(y_arr)
    return max_y - min_y

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:

    drowsy_frames = 0
    max_left = 0
    max_right = 0
    alert_start_time = None
    alert_duration = 4

    # Connect to the SQLite database
    conn = sqlite3.connect('eye_data.db')
    c = conn.cursor()

    # Create a table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS eye_data (
            timestamp REAL,
            max_left REAL,
            max_right REAL,
            len_left REAL,
            len_right REAL
        )
    ''')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            all_landmarks = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            right_eye = all_landmarks[RIGHT_EYE]
            left_eye = all_landmarks[LEFT_EYE]

            cv.polylines(frame, [left_eye], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [right_eye], True, (0, 255, 0), 1, cv.LINE_AA)

            len_left = open_len(right_eye)
            len_right = open_len(left_eye)

            if len_left > max_left:
                max_left = len_left

            if len_right > max_right:
                max_right = len_right

            cv.putText(img=frame, text='Max: ' + str(max_left) + ' Left Eye: ' + str(len_left), fontFace=0, org=(10, 30),
                       fontScale=0.5, color=(0, 255, 0))
            cv.putText(img=frame, text='Max: ' + str(max_right) + ' Right Eye: ' + str(len_right), fontFace=0,
                       org=(10, 50), fontScale=0.5, color=(0, 255, 0))

            if len_left <= int(max_left / 2) + 1 and len_right <= int(max_right / 2) + 1:
                drowsy_frames += 1
                if drowsy_frames == 1:
                    alert_start_time = time.time()
            else:
                drowsy_frames = 0
                alert_start_time = None

            if alert_start_time is not None and time.time() - alert_start_time >= alert_duration:
                cv.putText(img=frame, text='ALERT', fontFace=0, org=(10, 70), fontScale=0.7, color=(0, 0, 255))

            # Insert data into the database
            timestamp = time.time()
            c.execute('INSERT INTO eye_data VALUES (?, ?, ?, ?, ?)', (timestamp, max_left, max_right, len_left, len_right))
            conn.commit()

        cv.imshow('img', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    conn.close()
    cap.release()
    cv.destroyAllWindows()
