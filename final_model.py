import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 데이터 불러오기 및 전처리
five_folder = '0'
rock_folder = '1'
one1_folder = '2'
one2_folder = '3'
two1_folder = '4'
two2_folder = '5'
etc_folder = '6'

five_files = [f for f in os.listdir(five_folder) if f.endswith('.csv')]
rock_files = [f for f in os.listdir(rock_folder) if f.endswith('.csv')]
one1_files = [f for f in os.listdir(one1_folder) if f.endswith('.csv')]
one2_files = [f for f in os.listdir(one2_folder) if f.endswith('.csv')]
two1_files = [f for f in os.listdir(two1_folder) if f.endswith('.csv')]
two2_files = [f for f in os.listdir(two2_folder) if f.endswith('.csv')]
etc_files = [f for f in os.listdir(etc_folder) if f.endswith('.csv')]

rock_data = []
five_data = []
one1_data = []
one2_data = []
two1_data = []
two2_data = []
etc_data = []

# 파일들을 병합하는 과정
for file in rock_files:
    filepath = os.path.join(rock_folder, file)
    df = pd.read_csv(filepath, header=None, skiprows=1)
    rock_data.append(df.values)

for file in five_files:
    filepath = os.path.join(five_folder, file)
    df = pd.read_csv(filepath, header=None, skiprows=1)
    five_data.append(df.values)

for file in one1_files:
    filepath = os.path.join(one1_folder, file)
    df = pd.read_csv(filepath, header=None, skiprows=1)
    one1_data.append(df.values)

for file in one2_files:
    filepath = os.path.join(one2_folder, file)
    df = pd.read_csv(filepath, header=None, skiprows=1)
    one2_data.append(df.values)

for file in two1_files:
    filepath = os.path.join(two1_folder, file)
    df = pd.read_csv(filepath, header=None, skiprows=1)
    two1_data.append(df.values)

for file in two2_files:
    filepath = os.path.join(two2_folder, file)
    df = pd.read_csv(filepath, header=None, skiprows=1)
    two2_data.append(df.values)

for file in etc_files:
    filepath = os.path.join(etc_folder, file)
    df = pd.read_csv(filepath, header=None, skiprows=1)
    etc_data.append(df.values)

# 데이터를 numpy 배열로 변환
rock_data = np.array(rock_data)
one1_data = np.array(one1_data)
one2_data = np.array(one2_data)
two1_data = np.array(two1_data)
two2_data = np.array(two2_data)
five_data = np.array(five_data)
etc_data = np.array(etc_data)

# 데이터와 레이블 합치기
X = np.concatenate([five_data, rock_data, one1_data, one2_data, two1_data, two2_data, etc_data], axis=0)
y = np.concatenate([np.zeros(five_data.shape[0]), np.ones(rock_data.shape[0]), 2 * np.ones(one1_data.shape[0]),
                    3 * np.ones(one2_data.shape[0]),
                    4 * np.ones(two1_data.shape[0]), 5 * np.ones(two2_data.shape[0]), 6 * np.ones(etc_data.shape[0])],
                   axis=0)

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# CNN 모델 생성
conv_layer1 = layers.Conv1D(21, 2, padding='same')
conv_layer2 = layers.Conv1D(32, 2, padding='same', activation='relu')
conv_layer3 = layers.Conv1D(16, 2, activation='relu')

flatten = layers.Flatten()
dense1 = layers.Dense(32)
dense2 = layers.Dense(16)
dense3 = layers.Dense(7, activation='softmax')

inp = tf.keras.Input((21, 2))
x = conv_layer1(inp)
x = conv_layer2(x)
x = conv_layer3(x)
x = flatten(x)
x = dense1(x)
x = dense2(x)
out = dense3(x)

model = tf.keras.Model(inp, out, name='conv_model')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 콜백 정의
checkpoint_path = "best_model_pro.h5"
checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1,
                                       save_best_only=True, mode='max')

# 모델 훈련
history = model.fit(X_train, y_train, epochs=100, batch_size=40, validation_data=(X_test, y_test),
                    callbacks=[checkpoint])

# 저장된 최적의 모델 불러오기
saved_model = tf.keras.models.load_model(checkpoint_path)

# 모델 평가
loss, accuracy = saved_model.evaluate(X_test, y_test)
print("테스트 세트 손실:", loss)
print("테스트 세트 정확도:", accuracy)


# model = tf.keras.Sequential([
#     layers.LSTM(64, input_shape=(21, 2), return_sequences=True),
#     layers.Flatten(),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(7, activation='softmax')
# ])
# ## 모델부분만 바꿔서 돌리면됨
