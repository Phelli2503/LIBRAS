# 10. Avaliação usando Matriz de Confusão e Precisão
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# 6. Pré-processar dados e criar rótulos e recursos
from sklearn.model_selection import train_test_split  # Para dividir dados train/test
from keras.utils import to_categorical                 # Para one-hot encoding
    
# Construir e treinar rede neural LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf  

# 1. Import and Install Dependencies
import cv2 
import numpy as np
import time 
import os
from matplotlib import pyplot as plt
import mediapipe as mp 

# FUNÇÕES PARA O MEDIA PIPE
def mediapipe_detection(image, model):
    """
    Processa uma imagem com um modelo MediaPipe e retorna a imagem convertida e os resultados
    
    Args:
        image: Imagem de entrada no formato BGR (OpenCV padrão)
        model: Modelo MediaPipe configurado (Holistic, Hands, Pose, etc.)
    
    Returns:
        tuple: (imagem convertida de volta para BGR, resultados do MediaPipe)
    """
    # 1. Conversão de BGR (OpenCV) para RGB (MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Melhoria de performance: imagem somente-leitura
    image.flags.writeable = False
    
    # 3. Processamento pelo modelo MediaPipe
    results = model.process(image)
    
    # 4. Permite edição na imagem novamente
    image.flags.writeable = True
    
    # 5. Conversão de volta para BGR (OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

def draw_landmarks(image, results):
    """
    Desenha todos os landmarks (pontos-chave) e conexões detectados na imagem de entrada.
    ATENÇÃO: Corrigido FACE_CONNECTIONS para FACEMESH_TESSELATION
    """
    # Desenha os landmarks faciais (468 pontos)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    
    # Desenha os landmarks corporais (33 pontos)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    
    # Desenha os landmarks da mão esquerda (21 pontos)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    
    # Desenha os landmarks da mão direita (21 pontos)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    """
    Extrai e concatena os keypoints (pontos de referência) detectados pelo MediaPipe
    em um único array numpy para ser usado como entrada em modelos de aprendizado de máquina.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# [Mantenha todas as importações anteriores...]

# 2. Inicializar o MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 3. Definir ações e configurações
actions = np.array(['olá', 'obrigado', 'eu te amo'])
sequence_length = 30

# 7. Carregar o modelo treinado
model_path = 'action.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Treine o modelo primeiro.")

try:
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {str(e)}")
    exit()

# 11. Teste em Tempo Real (Versão Corrigida)
sequence = []
sentence = []
predictions = []
threshold = 0.4
cooldown_frames = 0
last_action = None
min_frames_for_detection = 15

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    static_image_mode=False
) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Processamento da imagem
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        # Extração de keypoints
        try:
            keypoints = extract_keypoints(results)
            keypoints = np.pad(keypoints, (0, 1662 - len(keypoints)))[:1662]  # Garante 1662 features
            
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
            
            if len(sequence) == sequence_length and cooldown_frames <= 0:
                input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                
                # Verificação adicional do shape
                if input_data.shape != (1, 30, 1662):
                    print(f"Shape inválido: {input_data.shape}. Esperado (1, 30, 1662)")
                    continue
                
                res = model.predict(input_data, verbose=0)[0]
                action_idx = np.argmax(res)
                confidence = res[action_idx]
                current_action = actions[action_idx]
                
                print("\nConfianças:")
                for name, conf in zip(actions, res):
                    print(f"{name}: {conf:.4f}")
                
                if (confidence > threshold and 
                    current_action != last_action and
                    len(predictions) > min_frames_for_detection and
                    np.all(np.array(predictions[-min_frames_for_detection:]) == action_idx)):
                    
                    sentence.append(current_action)
                    last_action = current_action
                    cooldown_frames = 20
                    print(f"\nAção detectada: {current_action} (Confiança: {confidence:.2f})")
                
                predictions.append(action_idx)
            
            cooldown_frames = max(0, cooldown_frames - 1)
            
            # Exibição
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            if len(sentence) > 0:
                cv2.putText(image, sentence[-1], (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Leitor de Libras', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            continue

cap.release()
cv2.destroyAllWindows()