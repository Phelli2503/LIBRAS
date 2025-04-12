# 6. Pré-processar dados e criar rótulos e recursos

from sklearn.model_selection import train_test_split  # Para dividir dados train/test
from keras.utils import to_categorical                 # Para one-hot encoding

# 1. Import and Install Depedencies
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

# 2. Keypoints using MP Holistic

# Inicializa o modelo Holistic do MediaPipe
# ATENÇÃO: Corrigido o nome para 'holistic' (com 's')
mp_holistic = mp.solutions.holistic  # Solução para detecção completa

# Inicializa as utilidades de desenho do MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Ferramentas para desenhar landmarks

# 4. Configurar pastas para coleção 
# Caminho para dados exportados, numpy e arrays  
""" 
Detecção de ação
Uma diferença fundamental entre a detecção de ações e outras tarefasa de visão computacional
é que uma sequencia de dados, em vez de um único quadro, é usada uma detecção.

 """
# Caminho para dados exportados
DATA_PATH = os.path.join('MP_Data')    

# Definir frases (ações) para salvar

# ================================================================================================================

# Ações que tentamos detectar / capturar
actions = np.array(['olá', 'obrigado', 'eu te amo'])

# ================================================================================================================

# 30 vídeos de dados
no_sequences = 30
# Os vídeos terão 30 quadros de duração
sequence_length = 30

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass



label_map = {label:num for num, label in enumerate(actions)}

# for action in actions:
#     for sequence in range(no_sequences):
#         for frame_num in range(sequence_length):
#             path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
#             if not os.path.exists(path):
#                 print(f"Arquivo ausente: {path}")
for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(path)
            print(f"{path} - Formato: {res.shape}")

# target_shape = (1662,)  # Ajuste conforme o formato correto dos seus dados

# for action in actions:
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
#             res = np.load(path)
            
#             # Ajustar tamanho se necessário
#             if res.shape != target_shape:
#                 res = np.resize(res, target_shape)
            
#             window.append(res)

