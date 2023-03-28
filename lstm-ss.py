import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 生成时间序列数据
def generate_sequence(length):
    return np.array([np.sin(0.1*i) for i in range(length)])

# 将时间序列数据转换成有监督学习问题的数据集
def create_dataset(sequence, look_back, look_forward):
    data, target = [], []
    for i in range(len(sequence)-look_back-look_forward):
        data.append(sequence[i:i+look_back])
        target.append(sequence[i+look_back:i+look_back+look_forward])
    return np.array(data), np.array(target)

# 设定参数
look_back = 10
look_forward = 5
epochs = 100
batch_size = 32

# 生成数据集
sequence = generate_sequence(1000)
X, y = create_dataset(sequence, look_back, look_forward)

# 将数据集划分为训练集和测试集
train_size = int(len(X) * 0.67)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 构建模型
model = Sequential()
model.add(LSTM(32, input_shape=(look_back, 1)))
model.add(Dense(look_forward))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

