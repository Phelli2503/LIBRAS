# 1 - IMPORTAR E INSTALAR DEPENDÊNCIAS
# Processamento de imagens/vídeos (OpenCV + NumPy)
import cv2 
import numpy as np
# Controle de tempo e arquivos (time, os)
import time 
import os
# Visualização de resultados (Matplotlib)
from matplotlib import pyplot as plt
# Detecção de landmarks corporais (MediaPipe)
import mediapipe as mp 

# 2 - PONTOS CHAVE USANDO MP HOLISTIC
mp_holistic = mp.solutions.holistic  # Modelo Holistic
mp_drawing = mp.solutions.drawing_utils  # Utilitários de desenho

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversão correta para RGB
    image.flags.writeable = False  # A imagem não é mais editável
    results = model.process(image)  # Realiza a detecção
    image.flags.writeable = True  # A imagem agora é editável
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte de volta para BGR para exibição
    return image, results

def draw_style_landmarks(image, results):
    # Desenhar conexões do rosto
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Desenhar conexões da pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))  
    # Desenhar conexões da mão esquerda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))  
    # Desenhar conexões da mão direita
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# 3 - EXTRAIR VALORES DOS PONTOS CHAVE
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue
                # Realizar detecções
                image, results = mediapipe_detection(frame, holistic)

                # Desenhar landmarks
                draw_style_landmarks(image, results)

                # Mostrar na tela
                cv2.imshow("OpenCV Feed", image)
                
                # Interromper de forma graciosa
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
cap.release()
cv2.destroyAllWindows()
