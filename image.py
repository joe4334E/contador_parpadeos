import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_face_mesh = mp.solutions.face_mesh

# Función para calcular la distancia entre dos puntos
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Función para detectar si el ojo está cerrado
def is_eye_closed(eye_landmarks):
    # Calcular la distancia entre los puntos superior e inferior del ojo
    top_to_bottom_distance = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    # Calcular la distancia entre los puntos izquierdo y derecho del ojo
    left_to_right_distance = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    # Calcular el aspect ratio del ojo
    aspect_ratio = top_to_bottom_distance / left_to_right_distance
    # El ojo se considera cerrado si el aspect ratio es menor que un umbral
    return aspect_ratio < 0.2

# Índices de los puntos de referencia de los ojos en la malla facial
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Malla Facial y Puntos de Ojos")
        
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.button = tk.Button(root, text="Cargar Imagen", command=self.load_image)
        self.button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        frame = cv.imread(file_path)
        img_h, img_w = frame.shape[:2]

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as face_mesh:

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extraer las coordenadas de los puntos de referencia del ojo derecho
                    right_eye_landmarks = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in RIGHT_EYE]
                    # Extraer las coordenadas de los puntos de referencia del ojo izquierdo
                    left_eye_landmarks = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in LEFT_EYE]

                    # Dibujar solo los puntos de referencia de los ojos
                    for x, y in right_eye_landmarks:
                        cv.circle(frame, (x, y), 1, (0, 255, 0), 1)
                    for x, y in left_eye_landmarks:
                        cv.circle(frame, (x, y), 1, (0, 255, 0), 1)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
            self.canvas.image = image

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()

