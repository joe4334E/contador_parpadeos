import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para calcular la distancia vertical entre los puntos extremos en el eje y
def open_len(arr):
    y_arr = []
    for _, y in arr:
        y_arr.append(y)
    min_y = min(y_arr)
    max_y = max(y_arr)
    return max_y - min_y

mp_face_mesh = mp.solutions.face_mesh

# Índices de landmarks faciales para la boca
MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84]

# Índices de landmarks faciales para los ojos
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Tiempo mínimo para detectar fatiga y bostezo
TIME_TO_DETECT_DROWSINESS = 3  # 3 segundos
TIME_TO_DETECT_YAWN = 5  # 5 segundos

# Manejador de la cámara web
cap = cv.VideoCapture(0)

# Parámetros de MediaPipe
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    # Contadores de fotogramas para detectar somnolencia y bostezos
    drowsy_frames = 0
    yawning_frames = 0

    # Contadores para contabilizar cuántas veces ocurre la fatiga y el bostezo
    drowsiness_count = 0
    yawn_count = 0

    # Altura máxima de la boca y altura máxima de los ojos
    max_mouth = 0
    max_left_eye = 0
    max_right_eye = 0

    start_time_drowsiness = time.time()
    start_time_yawn = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # Landmarks de la boca
            mouth_landmarks = all_landmarks[MOUTH]

            # Landmarks de los ojos
            right_eye_landmarks = all_landmarks[RIGHT_EYE]
            left_eye_landmarks = all_landmarks[LEFT_EYE]

            # Visualización de las máscaras de los landmarks de los ojos y la boca
            for landmark in mouth_landmarks:
                cv.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)

            for landmark in right_eye_landmarks:
                cv.circle(frame, tuple(landmark), 2, (0, 0, 255), -1)

            for landmark in left_eye_landmarks:
                cv.circle(frame, tuple(landmark), 2, (0, 0, 255), -1)

            # Dibujar la malla facial completa
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    cv.circle(frame, (x, y), 1, (255, 0, 0), -1)

            # Calcular la apertura de la boca
            mouth_opening = open_len(mouth_landmarks)

            # Calcular la altura de los ojos
            len_left_eye = open_len(left_eye_landmarks)
            len_right_eye = open_len(right_eye_landmarks)

            # Mantener la mayor distancia vertical de la apertura de la boca y de la altura de los ojos
            if mouth_opening > max_mouth:
                max_mouth = mouth_opening
                if len_left_eye > max_left_eye:
                    max_left_eye = len_left_eye
                if len_right_eye > max_right_eye:
                    max_right_eye = len_right_eye

                # Imprimir en pantalla la altura de la boca y de los ojos
                cv.putText(img=frame, text='Max Mouth: ' + str(max_mouth)  + ' Current: ' + str(mouth_opening), org=(10, 30), fontFace=0, fontScale=0.5, color=(0, 255, 0))
                cv.putText(img=frame, text='Max Left Eye: ' + str(max_left_eye)  + ' Current: ' + str(len_left_eye), org=(10, 50), fontFace=0, fontScale=0.5, color=(0, 255, 0))
                cv.putText(img=frame, text='Max Right Eye: ' + str(max_right_eye)  + ' Current: ' + str(len_right_eye), org=(10, 70), fontFace=0, fontScale=0.5, color=(0, 255, 0))

                # Condición para detectar somnolencia: si los ojos están medio cerrados
                if (len_left_eye <= int(max_left_eye / 2) + 1 and len_right_eye <= int(max_right_eye / 2) + 1):
                    drowsy_frames += 1
                else:
                    drowsy_frames = 0

                # Si el conteo de fotogramas de somnolencia supera un cierto umbral, mostrar una alerta
                if drowsy_frames > 20:  # Si hay 20 fotogramas seguidos con signos de somnolencia
                    current_time_drowsiness = time.time()
                    if current_time_drowsiness - start_time_drowsiness >= TIME_TO_DETECT_DROWSINESS:  # Si han pasado TIME_TO_DETECT_DROWSINESS segundos desde la última detección de fatiga
                        start_time_drowsiness = current_time_drowsiness
                        drowsiness_count += 1
                        cv.putText(img=frame, text=f'Drowsiness detected: {drowsiness_count}', org=(200, 200), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

                # Condición para detectar bostezos: si la boca está abierta más de cierto umbral
                if mouth_opening > int(max_mouth / 2) + 1:
                    yawning_frames += 1
                else:
                    yawning_frames = 0

                # Si el conteo de fotogramas de bostezo supera un cierto umbral, mostrar una alerta
                if yawning_frames > 20:  # Si hay 20 fotogramas seguidos con signos de bostezo
                    current_time_yawn = time.time()
                    if current_time_yawn - start_time_yawn >= TIME_TO_DETECT_YAWN:  # Si han pasado TIME_TO_DETECT_YAWN segundos desde la última detección de bostezo
                        start_time_yawn = current_time_yawn
                        yawn_count += 1
                        cv.putText(img=frame, text=f'Yawn detected: {yawn_count}', org=(200, 250), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        
        # Mostrar siempre la imagen de la cámara
        cv.imshow('img', frame)

        # Mantener la ventana abierta hasta que se presione 'q'
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

