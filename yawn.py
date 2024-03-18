import cv2 as cv
import numpy as np
import mediapipe as mp

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

# Manejador de la cámara web
cap = cv.VideoCapture(0)

# Parámetros de MediaPipe
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    # Contador de fotogramas para detectar bostezos
    yawning_frames = 0

    # Altura máxima de la boca
    max_mouth = 0

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

            # Dibujar los landmarks de la boca en el marco
            cv.polylines(frame, [mouth_landmarks], True, (0, 255, 0), 1, cv.LINE_AA)

            # Calcular la apertura de la boca
            mouth_opening = open_len(mouth_landmarks)

            # Mantener la mayor distancia vertical de la apertura de la boca
            if mouth_opening > max_mouth:
                max_mouth = mouth_opening

            # Imprimir en pantalla la altura de la boca
            cv.putText(img=frame, text='Max Mouth: ' + str(max_mouth)  + ' Current: ' + str(mouth_opening), org=(10, 30), fontFace=0, fontScale=0.5, color=(0, 255, 0))

            # Condición: si la boca está abierta más de cierto umbral, contar fotogramas
            if mouth_opening > int(max_mouth / 2) + 1:
                yawning_frames += 1
            else:
                yawning_frames = 0

            # Si el conteo de fotogramas de bostezo supera un cierto umbral, mostrar una alerta
            if yawning_frames > 20:
                cv.putText(img=frame, text='Yawning Alert', org=(200, 300), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

