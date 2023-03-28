import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 生成多元时间序列数据
def generate_data(sequence_length=10000):
    # 生成正弦和余弦函数数据
    x = np.arange(sequence_length)
    sin_data = np.sin(2*np.pi*x/sequence_length)
    cos_data = np.cos(2*np.pi*x/sequence_length)
    
    # 组织多元时间序列数据
    data = np.empty((sequence_length, 2))
    data[:,0] = sin_data
    data[:,1] = cos_data
        
    return data

def create_dataset(sequence, look_back, look_forward):
    data, target = [], []
    for i in range(len(sequence)-look_back-look_forward):
        data.append(sequence[i:i+look_back])
        target.append(sequence[i+look_back:i+look_back+look_forward,0])
    return np.array(data), np.array(target)

# 设置参数
look_back = 140
look_forward = 8
input_shape = (look_back, 2)

# 生成数据
seq = generate_data()
X, y = create_dataset(seq, look_back, look_forward)

# 构建模型
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dense(look_forward))
model.compile(loss='mse', optimizer='adam')

# 训练模型
model.fit(X, y, epochs=100, batch_size=32)

