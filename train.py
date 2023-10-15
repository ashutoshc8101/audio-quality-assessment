from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from model.keras.transformer import transformer
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kendalltau
import time

df = pd.read_csv('./dataset/original.csv')

df = df.drop(['Unnamed: 0'], axis = 1)
df = df.dropna()

np.random.seed(42)
size = 415
df_read = df.copy()
df1 = df_read.sample(size)
df_read = df_read.drop(df1.index)
df2 = df_read.sample(size)
df_read = df_read.drop(df2.index)
df3 = df_read.sample(size)
df_read = df_read.drop(df3.index)
df4 = df_read.sample(size)
df_read = df_read.drop(df4.index)
df5 = df_read.copy()

q = list(df1.index) + list(df2.index) + list(df3.index) + list(df4.index) + list(df5.index)

train = pd.concat([df1,df2,df3,df4], axis=0)
train = train.dropna()

train = pd.concat([df1,df2,df3,df4], axis=0)
test = df5.copy()

Y_train = np.array(train['class'])
X_train = np.array(train.drop(['0','5','9','108', 'class'],axis=1))
X_train = X_train.reshape(X_train.shape[0], 1 , X_train.shape[1])

Y_val = np.array(test['class'])
X_val = np.array(test.drop(['0','5','9','108', 'class'],axis=1))
X_val = X_val.reshape(X_val.shape[0], 1 , X_val.shape[1])
NUM_LAYERS = 4

D_MODEL = X_train.shape[2]
NUM_HEADS = 4
UNITS = 2048
DROPOUT = 0.1
TIME_STEPS = X_train.shape[1]
OUTPUT_SIZE = 1
batch_size = 64

model = transformer(
  time_steps = TIME_STEPS,
  num_layers = NUM_LAYERS,
  units = UNITS,
  d_model = D_MODEL,
  num_heads = NUM_HEADS,
  dropout = DROPOUT,
  output_size = OUTPUT_SIZE,
  projection = 'linear'
)

model.compile(optimizer=tf.keras.optimizers.Adam(0.000003), loss='mse')

history = model.fit(X_train,Y_train, epochs=1000, validation_data=(X_val, Y_val))

st = time.time()
p1 = np.array(model(X_val)).flatten()
end = time.time()

print((end - st)/len(p1))

p2 = np.array(model(X_train)).flatten()

print(X_train.shape)
print(X_val.shape)

print("Training Correlation")
print(np.corrcoef(p2, Y_train), stats.spearmanr(p2, Y_train), kendalltau(p2,Y_train).correlation)

print("Validation Correlation")
print(np.corrcoef(p1, Y_val)[1][0],
      stats.spearmanr(p1, Y_val).correlation,
      kendalltau(p1,Y_val).correlation)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss.jpg")