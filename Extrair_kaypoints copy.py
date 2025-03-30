''' Gameplan

GOAL: Real time sign language detection using sequences

1. Extract holistic keypoints

    Collect keypoints from mediapipe holistc

2. Train an LSTM DL Model

    Train a deep neural network with LSTM layers for sequences

3. Make real time predictions using sequences

    Peform real time sign language detection using OpenCV

 '''

# 1. Import and Install Depedencies
#Processamento de imagens/vídeos (OpenCV + NumPy).
import cv2 
import numpy as np
#Controle de tempo e arquivos (time, os).
import time 
import os
#Visualização de resultados (Matplotlib).
from matplotlib import pyplot as plt
#Detecção de landmarks corporais (MediaPipe).
import mediapipe as mp 

# 2. Keypoints using MP Holistic

# Inicializa o modelo Holistic do MediaPipe
# Holistic é um modelo que detecta e rastreia:
# - Pose (postura corporal com 33 landmarks)
# - Mãos (21 landmarks para cada mão)
# - Rosto (468 landmarks faciais)
mp_holistic = mp.solutions.holistic

# Inicializa as utilidades de desenho do MediaPipe
# Essas funções são usadas para visualizar os landmarks detectados:
# - Desenha conexões entre landmarks (linhas)
# - Desenha os próprios landmarks (pontos)
# - Permite customização de cores e espessuras
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Processa uma imagem com um modelo MediaPipe e retorna a imagem convertida e os resultados
    
    Args:
        image: Imagem de entrada no formato BGR (OpenCV padrão)
        model: Modelo MediaPipe configurado (Holistic, Hands, Pose, etc.)
    
    Returns:
        tuple: (imagem convertida de volta para BGR, resultados do MediaPipe)
    """
    
    # 1. Conversão de BGR (formato OpenCV) para RGB (formato que MediaPipe espera)
    # MediaPipe requer imagens em formato RGB para processamento correto
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Melhoria de performance: marca a imagem como não-editável
    # Isso permite que o MediaPipe otimize o processamento (a imagem não será copiada)
    image.flags.writeable = False
    
    # 3. Processamento pelo modelo MediaPipe
    # Extrai todos os landmarks (pontos-chave) e informações de detecção
    results = model.process(image)
    
    # 4. Marca a imagem como editável novamente
    # Permite que desenhemos os resultados ou façamos outras modificações
    image.flags.writeable = True
    
    # 5. Conversão de volta para BGR (formato OpenCV) para visualização correta
    # Necessário porque cv2.imshow() espera imagens no formato BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

# Função para desenhar na camera os landmarks
def draw_landmarks(image, results):
    """
    Desenha todos os landmarks (pontos-chave) e conexões detectados na imagem de entrada.
    
    Args:
        image: Imagem de entrada no formato BGR (padrão OpenCV)
        results: Objeto de resultados do MediaPipe contendo os landmarks detectados
    """
    
    # Desenha os landmarks e conexões do rosto (468 pontos)
    # Só desenha se landmarks faciais foram detectados
    mp_drawing.draw_landmarks(
        image, 
        results.face_landmarks, 
        mp_holistic.FACE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,110,10)),   # Cor das conexões (BGR - verde)
        mp_drawing.DrawingSpec(color=(80,256,121)))  # Cor dos pontos (BGR - verde claro)
    
    # Desenha os landmarks e conexões da postura (33 pontos)
    # Só desenha se landmarks corporais foram detectados
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10)),    # Cor das conexões (BGR - vermelho escuro)
        mp_drawing.DrawingSpec(color=(80,44,121)))   # Cor dos pontos (BGR - vermelho claro)
    
    # Desenha os landmarks e conexões da mão esquerda (21 pontos)
    # Só desenha se landmarks da mão esquerda foram detectados
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76)),   # Cor das conexões (BGR - roxo)
        mp_drawing.DrawingSpec(color=(121,44,250)))  # Cor dos pontos (BGR - roxo claro)
    
    # Desenha os landmarks e conexões da mão direita (21 pontos)
    # Corrigido o typo de 'righ_hand_landmarks' para 'right_hand_landmarks'
    # Só desenha se landmarks da mão direita foram detectados
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66)),  # Cor das conexões (BGR - laranja)
        mp_drawing.DrawingSpec(color=(245,66,230)))  # Cor dos pontos (BGR - rosa)

# Inicializa a captura de vídeo da câmera padrão (normalmente a webcam)
# O argumento '0' indica que queremos usar a primeira câmera disponível
cap = cv2.VideoCapture(0)

# Definir modelo do MediaPipe
# Inicializa o modelo Holistic do MediaPipe dentro de um contexto 'with' 
# (gerenciamento automático de recursos)
with mp_holistic.Holistic(
    # Parâmetros de configuração:
    min_detection_confidence=0.5,  # Confiança mínima (50%) para considerar uma detecção válida
    min_tracking_confidence=0.5    # Confiança mínima (50%) para continuar rastreando objetos detectados
) as holistic:  # Cria uma instância do modelo chamada 'holistic'

    # Entra em um loop enquanto a câmera estiver aberta e funcionando corretamente
    while cap.isOpened():
        # Lê um frame do vídeo da câmera
        # 'ret' é um booleano que indica se o frame foi lido com sucesso
        # 'frame' contém a imagem capturada
        ret, frame = cap.read()

        # fazer detecções
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Exibe o frame capturado em uma janela chamada 'Janela OpenCV'
        cv2.imshow('Janela OpenCV', frame)
        
        # Aguarda 10ms por uma tecla ser pressionada e verifica se foi a tecla 'q'
        # 0xFF é usado para máscara de bits (necessário em alguns sistemas)
        # Se 'q' for pressionado, sai do loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        ''' VALORES DE REFERÊNCIA
        Os modelos de pontos de referência do rosto e da mão não retornarão valores se nada for detectado
        O modelo de pose retornará pontos de referência, mas o valor de visibilidade dentro de cada ponto de referência será baixo

            DRAW_LANDMARKS
         A função draw_markings não retorna a imagem, mas aplica as visualizações de landmarks à imagem atual no local
        '''
        draw_landmarks(frame,results)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    # Libera os recursos da câmera (importante para evitar travamentos)
    cap.release()

    # Fecha todas as janelas abertas pelo OpenCV
    cv2.destroyAllWindows()



