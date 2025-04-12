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

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

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
            mp_holistic.FACEMESH_TESSELATION,  # Conexões faciais corretas
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

# 3. Extrair valores keypoint
def extract_keypoints(results):
    """
    Extrai e concatena os keypoints (pontos de referência) detectados pelo MediaPipe
    em um único array numpy para ser usado como entrada em modelos de aprendizado de máquina.
    
    Args:
        results: Objeto de resultados retornado pelo MediaPipe Holistic
        
    Returns:
        np.array: Array concatenado contendo todos os keypoints (pose, rosto, mãos)
    """
    
    # Extrai os keypoints da pose corporal (33 landmarks)
    # Cada landmark tem coordenadas (x,y,z) e valor de visibilidade (4 valores por landmark)
    # Se não houver detecção, preenche com zeros (33 landmarks * 4 valores)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extrai os keypoints faciais (468 landmarks)
    # Cada landmark tem coordenadas (x,y,z) (3 valores por landmark)
    # Se não houver detecção, preenche com zeros (468 landmarks * 3 valores)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Extrai os keypoints da mão esquerda (21 landmarks)
    # Cada landmark tem coordenadas (x,y,z) (3 valores por landmark)
    # Se não houver detecção, preenche com zeros (21 landmarks * 3 valores)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Extrai os keypoints da mão direita (21 landmarks)
    # Cada landmark tem coordenadas (x,y,z) (3 valores por landmark)
    # Se não houver detecção, preenche com zeros (21 landmarks * 3 valores)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatena todos os arrays em um único array 1D
    # Ordem: pose (132 valores) + face (1404 valores) + mão esquerda (63 valores) + mão direita (63 valores)
    # Total: 132 + 1404 + 63 + 63 = 1662 valores (igual mencionado no comentário original)
    return np.concatenate([pose, face, lh, rh])

# 5 Coletar valores keypoints para treinar e testar
# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Configuração do modelo Holistic
# ATENÇÃO: Adicionado parâmetro refine_face_landmarks para melhor detecção facial
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_face_landmarks=True  # Melhora landmarks faciais
) as holistic:

    # while cap.isOpened():
    # Percorrer a ação
    '''Intervalos de coleta
    Fazer pausas entre cada sequência
    permite que você se recomponha e
    se reposicione para coletar a ação
    do início ao fim (# Aplicar lógica de coleção)
    '''
    for action in actions:
        # percorrer sequências, também conhecidas como vídeos
        for sequence in range(no_sequences):
            # percorrer a duração do vídeo, também conhecida como duração da sequência
            for frame_num in range(sequence_length):

                # read feed
                ret, frame = cap.read()
                # if not ret:
                #     print("Ignoring empty camera frame.")
                #     continue
                    
                # Processa a imagem
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                # Desenha os landmarks - AGORA na imagem processada
                draw_landmarks(image, results)

                # Aplicar lógica de coleção
                # Verifica se é o primeiro frame da coleta (frame número 0)
                if frame_num == 0:
                    # Mostra um texto grande centralizado indicando o início da coleta
                    cv2.putText(
                        image,                      # Imagem onde o texto será desenhado
                        'Iniciando coleta',         # Texto a ser exibido
                        (120, 200),                 # Posição (x,y) do texto na imagem
                        cv2.FONT_HERSHEY_SIMPLEX,   # Tipo de fonte
                        1,                          # Escala/tamanho da fonte
                        (0, 255, 0),                # Cor do texto (verde em BGR)
                        4,                          # Espessura da fonte
                        cv2.LINE_AA                 # Tipo de linha (anti-aliased)
                    )
                    
                    # Mostra informações sobre qual ação está sendo gravada
                    cv2.putText(
                        image,
                        'Coletando frames para {} Video Número {}'.format(action, sequence),  # Texto formatado
                        (15, 12),                  # Posição no canto superior esquerdo
                        cv2.FONT_HERSHEY_SIMPLEX,   # Tipo de fonte
                        0.5,                       # Escala menor que o texto principal
                        (0, 0, 255),               # Cor vermelha (BGR)
                        1,                         # Espessura mais fina
                        cv2.LINE_AA
                    )
                    
                    # Pausa a execução por 2000ms (2 segundos) para dar tempo do usuário se preparar
                    cv2.waitKey(2000)

                    ''' 
                    Encutar Pausas
                    Para encurtar o intervalo entre sequencias coletadas
                    altere o valor em cv2.waitTime() para um valor menor.
                    Para aumentar o tempo, aumente o valor.
                    '''

                # Para todos os outros frames (não é o primeiro frame)
                else:
                    # Mostra apenas a informação da ação e número do vídeo (sem o texto "Iniciando coleta")
                    cv2.putText(
                        image,
                        'Coletando frames para {} Video Número {}'.format(action, sequence),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )
                
                # Exportando Keypoints
                keypoints = extract_keypoints(results)
                
                # Verificar formato dos dados
                print(f"Tamanho do array de keypoints: {keypoints.shape}")

                # DESCOMENTAR
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                
                # Mostra o resultado
                cv2.imshow('Leitor de Libras', image)
                
                # Finaliza com 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # 3. Extrair valores keypoint
        # Função para extrair
        print(extract_keypoints(results).shape)

        # pose = []

        # # for res in results.pose_landmarks.landmark:
        # #     test = np.array([res.x, res.y, res.z, res.visibility])
        # #     # print(test)
        # #     pose.append(test)

        # ''' Dados de entrada
        # Os dados de entrada usados ​​para este modelo de detecção de ação são uma série de 30 matrizes, cada uma contendo 1.662 valores (30, 1.662).

        # Cada uma das 30 matrizes representa os valores de referência (1662 valores) de um único quadro. '''        
 
        # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        
        # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        # rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

        # print(face)    


cap.release()
cv2.destroyAllWindows()