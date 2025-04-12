# 6. Pré-processar dados e criar rótulos e recursos
from sklearn.model_selection import train_test_split  # Para dividir dados train/test
from keras.utils import to_categorical                 # Para one-hot encoding
    
    # Construir e treinar rede neural LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# 1. Import and Install Dependencies
import cv2 
import numpy as np
import time 
import os
from matplotlib import pyplot as plt
import mediapipe as mp 

# 2. Inicializar o MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 3. Definir caminhos e configurações
DATA_PATH = os.path.join('MP_Data')
# actions = np.array(['olá', 'okay' ,'obrigado', 'eu te amo'])
actions = np.array(['olá', 'obrigado', 'eu te amo'])
no_sequences = 30
sequence_length = 30

# Criar diretórios se não existirem
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 4. Criar rótulos e mapear ações
label_map = {label: num for num, label in enumerate(actions)}

# 5. Processar os dados
sequences, labels = [], []
EXPECTED_SHAPE = 1662  # Definir o tamanho esperado dos vetores

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            
            # Garantir que todos os vetores tenham o tamanho correto
            if res.shape[0] < EXPECTED_SHAPE:
                res = np.pad(res, (0, EXPECTED_SHAPE - res.shape[0]))
            elif res.shape[0] > EXPECTED_SHAPE:
                res = res[:EXPECTED_SHAPE]

            window.append(res)
        
        sequences.append(window)
        labels.append(label_map[action])

# Converter para numpy arrays
# sequences = np.array(sequences)
# print(np.array(labels).shape)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# print(y_test.shape)

# 6. Exibir informações
# print(np.array(sequences).shape)
# print("Formato final dos dados:", sequences.shape)
# print("Formato dos rótulos:", labels.shape)

# 7 Construir e treinar rede neural LSTM
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

'''Número de Épocas
2000 épocas pode ser um número alto
para treinamento com uma quantidade tão 
baixa de dados.

Sinta-se à vontade para interromper o treinamento 
mais cedo se a precisão for aceitável e a perda tiver parado 
de diminuir consistentemente.
'''


# 8. Fazer Previsões

res = model.predict(x_test)

print(actions[np.argmax(res[0])])
print(np.argmax(y_test[0]))
print('==========================')

print(actions[np.argmax(res[2])])
print(np.argmax(y_test[2]))
print('==========================')

print(actions[np.argmax(res[4])])
print(np.argmax(y_test[4]))


# 9. Salvar Pesos
model.save('action.h5')

# model.load_weights('action.h5')  
