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
# Inicializa o modelo Holistic do MediaPipe
mp_holistic = mp.solutions.holistic  # Solução para detecção completa
# Inicializa as utilidades de desenho do MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Ferramentas para desenhar landmarks

# ========= FUNÇÕES ===================================================================
def mediapipe_detection(image, model):
    
#     Processa uma imagem com um modelo MediaPipe e retorna a imagem convertida e os resultados
    
#     Args:
#         image: Imagem de entrada no formato BGR (OpenCV padrão)
#         model: Modelo MediaPipe configurado (Holistic, Hands, Pose, etc.)
    
#     Returns:
#         tuple: (imagem convertida de volta para BGR, resultados do MediaPipe)

# 1.1 Conversão de BGR (OpenCV) para RGB (MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversão correta para RGB
# 1.2 Melhoria de performance: imagem somente-leitura
    image.flags.writeable = False  # A imagem não é mais editável
# 1.3 Processamento pelo modelo MediaPipe
    results = model.process(image)  # Realiza a detecção
# 1.4 Permite edição na imagem novamente
    image.flags.writeable = True  # A imagem agora é editável
# 1.5 Conversão de volta para BGR (OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte de volta para BGR para exibição
   
    return image, results

def draw_style_landmarks(image, results):
#    Desenha todos os landmarks (pontos-chave) e conexões detectados na imagem de entrada.

    # Desenhar conexões do rosto (468 pontos)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Desenhar conexões da pose (33 pontos)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))  
    # Desenhar conexões da mão esquerda (21 pontos)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))  
    # Desenhar conexões da mão direita (21 pontos)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# 3 - EXTRAIR VALORES DOS PONTOS CHAVE

#     Extrai e concatena os keypoints (pontos de referência) detectados pelo MediaPipe
#     em um único array numpy para ser usado como entrada em modelos de aprendizado de máquina.
    
#     Args:
#         results: Objeto de resultados retornado pelo MediaPipe Holistic
        
#     Returns:
#         np.array: Array concatenado contendo todos os keypoints (pose, rosto, mãos)

def extract_keypoints(results):

#     Extrai e concatena os keypoints (pontos de referência) detectados pelo MediaPipe
#     em um único array numpy para ser usado como entrada em modelos de aprendizado de máquina.
    
#     Args:
#         results: Objeto de resultados retornado pelo MediaPipe Holistic contendo:
#                 - pose_landmarks: Pontos de referência corporais (33 landmarks)
#                 - face_landmarks: Pontos de referência faciais (468 landmarks)
#                 - left_hand_landmarks: Pontos de referência da mão esquerda (21 landmarks)
#                 - right_hand_landmarks: Pontos de referência da mão direita (21 landmarks)
    
#     Returns:
#         np.array: Array 1D concatenado contendo todos os keypoints no formato:
#                   [pose_points, face_points, left_hand_points, right_hand_points]
 
    
    # Extrai os keypoints da pose corporal (33 landmarks)
    # Cada landmark tem 4 valores: x, y, z coordenadas + visibilidade
    # Se nenhum landmark for detectado, preenche com zeros (33 landmarks * 4 valores = 132 zeros)
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                    for res in results.pose_landmarks.landmark]).flatten() \
                    if results.pose_landmarks else np.zeros(132)
    
    # Extrai os keypoints faciais (468 landmarks)
    # Cada landmark tem 3 valores: x, y, z coordenadas
    # Se nenhum landmark for detectado, preenche com zeros (468 landmarks * 3 valores = 1404 zeros)
    face = np.array([[res.x, res.y, res.z] 
                    for res in results.face_landmarks.landmark]).flatten() \
                    if results.face_landmarks else np.zeros(1404)
    
    # Extrai os keypoints da mão esquerda (21 landmarks)
    # Cada landmark tem 3 valores: x, y, z coordenadas
    # Se nenhum landmark for detectado, preenche com zeros (21 landmarks * 3 valores = 63 zeros)
    lh = np.array([[res.x, res.y, res.z] 
                  for res in results.left_hand_landmarks.landmark]).flatten() \
                  if results.left_hand_landmarks else np.zeros(21*3)
    
    # Extrai os keypoints da mão direita (21 landmarks)
    # Cada landmark tem 3 valores: x, y, z coordenadas
    # Se nenhum landmark for detectado, preenche com zeros (21 landmarks * 3 valores = 63 zeros)
    rh = np.array([[res.x, res.y, res.z] 
                  for res in results.right_hand_landmarks.landmark]).flatten() \
                  if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatena todos os arrays em um único array 1D na ordem:
    # 1. Pose points (132 valores)
    # 2. Face points (1404 valores)
    # 3. Left hand points (63 valores)
    # 4. Right hand points (63 valores)
    # Total: 132 + 1404 + 63 + 63 = 1662 valores
    return np.concatenate([pose, face, lh, rh])

# 4 - CONFIGURAR PASTAS PARA COLETA
DATA_PATH = os.path.join('MP_Data')

# Ações que tentamos detectar
actions = np.array(['olá', 'obrigado', 'eu te amo'])
# actions = np.array(['hello', 'thanks', 'iloveyou'])
# Quadros 
no_sequences = 30  # Número de sequências
sequence_length = 30  # Comprimento dos vídeos

# 5 - COLETAR VALORES DOS PONTOS CHAVE PARA TREINAMENTO E TESTE
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

for action in actions:
    # Cria pastas para ações
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)  

# Adicionando a criação de subpastas para sequências
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))  # Corrigido para usar os.path.join
        except:
            pass


cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break  # Se não conseguir ler o frame, sair do loop

                # Realizar detecções
                image, results = mediapipe_detection(frame, holistic)

                # Desenhar landmarks
                draw_style_landmarks(image, results)

                # Lógica de espera
                if frame_num == 0:
                    cv2.putText(image, 'INICIANDO COLETA', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Coletando frames para {action} Sequencia {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Coletando frames para {action} Sequencia {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Exportar novos pontos chave
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Mostrar na tela
                cv2.imshow("OpenCV Feed", image)
                
                # Interromper de forma graciosa
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
cap.release()
cv2.destroyAllWindows()
