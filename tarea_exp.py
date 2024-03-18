import cv2 as cv
import numpy as np
import mediapipe as mp
import pygame
import os

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

# Inicializar pygame para reproducir sonidos
pygame.mixer.init()

# Cargar archivos de sonido
fatigue_sound = pygame.mixer.Sound('alarm1.wav')
yawn_sound = pygame.mixer.Sound('wake_up.wav')

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
    drowsy_frames = 60  # Cambiado a 60 fotogramas (equivalente a 3 segundos a 20 FPS)
    yawning_frames = 120  # Cambiado a 120 fotogramas (equivalente a 6 segundos a 20 FPS)

    # Altura máxima de la boca y altura máxima de los ojos
    max_mouth = 0
    max_left_eye = 0
    max_right_eye = 0

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

            for landmark in mouth_landmarks:
                cv.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)

            for landmark in right_eye_landmarks:
                cv.circle(frame, tuple(landmark), 2, (0, 0, 255), -1)

            for landmark in left_eye_landmarks:
                cv.circle(frame, tuple(landmark), 2, (0, 0, 255), -1)
            # Dibujar la malla facial completa
            #for face_landmarks in results.multi_face_landmarks:
            #    for landmark in face_landmarks.landmark:
            #       x = int(landmark.x * img_w)
            #       y = int(landmark.y * img_h)
            #       cv.circle(frame, (x, y), 1, (255, 0, 0), -1)

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

            # Dibujar los textos en posiciones adecuadas
            cv.putText(img=frame, text='Max Mouth: ' + str(max_mouth)  + ' Current: ' + str(mouth_opening), org=(10, 30), fontFace=0, fontScale=0.5, color=(0, 255, 0))
            cv.putText(img=frame, text='Max Left Eye: ' + str(max_left_eye)  + ' Current: ' + str(len_left_eye), org=(10, 50), fontFace=0, fontScale=0.5, color=(0, 255, 0))
            cv.putText(img=frame, text='Max Right Eye: ' + str(max_right_eye)  + ' Current: ' + str(len_right_eye), org=(10, 70), fontFace=0, fontScale=0.5, color=(0, 255, 0))
            # Condición para detectar somnolencia: si los ojos están medio cerrados
            if (len_left_eye <= int(max_left_eye / 2) + 1 and len_right_eye <= int(max_right_eye / 2) + 1):
                drowsy_frames -= 1
                if drowsy_frames == 0:
                    fatigue_sound.play()
                    cv.putText(img=frame, text='Estás cansado?', org=(img_w // 2 - 100, img_h // 2), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
                    drowsy_frames = 60  # Reiniciar contador si no se detecta somnolencia
            else:
                drowsy_frames = 60  # Reiniciar contador si no se detecta somnolencia

            # Condición para detectar bostezos: si la boca está abierta más de cierto umbral
            if mouth_opening > int(max_mouth / 2) + 1:
                yawning_frames -= 1
                if yawning_frames == 0:
                    yawn_sound.play()
                    cv.putText(img=frame, text='No bostees, descansa', org=(img_w // 2 - 130, img_h // 2), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                    yawning_frames = 120  # Reiniciar contador si no se detecta bostezo
            else:
                yawning_frames = 120  # Reiniciar contador si no se detecta bostezo

        # Mostrar siempre la imagen de la cámara
        cv.imshow('img', frame)

        # Mantener la ventana abierta hasta que se presione 'q'
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

